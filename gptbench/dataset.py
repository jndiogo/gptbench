"""
Dataset classes derived from torch.utils.data.Dataset
"""

import os, sys, copy, array, random, math, pdb

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

import tiktoken

from .utils import is_utf8, consumme_decode_utf8



# -----------------------------------------------------------------------------

''' 
Don't store tiktoken in object, because tokenizer is not pickable: https://github.com/huggingface/datasets/issues/5769

DataLoader with num_workers > 0 will error:
TypeError: cannot pickle 'builtins.CoreBPE' object
'''
GPT2_ENC = tiktoken.get_encoding("gpt2")


class GPT2TokensDataset(Dataset):
    """
    Emits sequences of block_size GPT2 tokens randomly sampled from a dataset
    """

    @staticmethod
    def load_train_val_datasets(train_path, val_path_or_train_split,
                                block_size, 
                                repeat_if_needed=False,
                                verbose=True):
        """ returns train_dataset, val_dataset - val dataset can be None """

        data = GPT2TokensDataset.load_data(data_path=train_path, verbose=verbose)

        if isinstance(val_path_or_train_split, str): # val from path
            train = GPT2TokensDataset(block_size, data=data, repeat_if_needed=repeat_if_needed, verbose=verbose)
            val = GPT2TokensDataset(block_size, data_path=val_path_or_train_split, repeat_if_needed=repeat_if_needed, verbose=verbose)

        else: # split from train
            assert isinstance(val_path_or_train_split, float), "val_path_or_train_split can be of str (path) or float (train_split) types"

            assert val_path_or_train_split > 0. and val_path_or_train_split <= 1., "0 < train split <= 1"

            # handle dummy dataset split:
            split_index = int(len(data) * val_path_or_train_split)
            if split_index == 0 and len(data) == 1: # ensure a train dataset with one entry
                split_index=1

            train = GPT2TokensDataset(block_size, data=data[:split_index], repeat_if_needed=repeat_if_needed, verbose=verbose)

            if split_index > 0 and split_index < len(data):
                val = GPT2TokensDataset(block_size, data=data[split_index:], repeat_if_needed=repeat_if_needed, verbose=verbose)
            else:
                val = None

        return train, val



    @staticmethod
    def load_data(data=None, data_path=None, verbose=False):

        enc = GPT2_ENC # tiktoken.get_encoding("gpt2")

        if data_path is not None:
            with open(data_path, 'rb') as f:
                data = f.read()

        if data is None: # dummy dataset with size 1 and first token repeated
            if verbose:
                print("Dataset: dummy 0 tokens")
            data = [0]

        elif is_utf8(data): # tokenize
            if verbose:
                print("Dataset: encoding utf-8 to tokens")
            text = data.decode(encoding='utf-8', errors='strict')
            data = enc.encode(text, allowed_special={"<|endoftext|>"}) # WAS encode_ordinary(text)
            if data[-1] != enc.eot_token:
                data.append(enc.eot_token) # ensure finalizing '<|endoftext|>'

        else: # must be list of uint16 tokens 
            if verbose:
                print("Dataset: loading uint16 tokens")
            data = array.array('H', data).tolist()

        return data



    def __init__(self, block_size, data=None, data_path=None, repeat_if_needed=False, verbose=True):
        """ data is a np.array(dtype=np.uint16) 
        if both data and data_path are None, a dummy dataset with token id=0 is created
        repeat_if_needed: if data size is less than block_size+1, repeat existing data as many times as needed (complete repeats only)
        """

        self.data = GPT2TokensDataset.load_data(data=data, data_path=data_path, verbose=verbose)
        # self.data is now encoded as uint16 tokens

        if len(self.data) < block_size+1 and repeat_if_needed:
            times = int(math.ceil((block_size+1) / len(self.data)))

            if verbose:
                print(f"Expanding initial dataset size of {len(self.data)} (less than block_size+1) by {times} times to size of {times*len(self.data)}")
            self.data = self.data * times

        assert len(self.data) > block_size, f"data length ({len(self.data)}) must be greater than block_size({block_size})"

        enc = GPT2_ENC # tiktoken.get_encoding("gpt2")

        self.vocab_size = enc.n_vocab
        self.block_size = block_size

        self.data_len = len(self.data) - self.block_size

        self.bufd_decode_list = [] # partial utf-8 undecoded byte list


    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return self.data_len

    def is_text_valid(self, text):
        try:
            enc = GPT2_ENC # tiktoken.get_encoding("gpt2")
            enc.encode(text, allowed_special={"<|endoftext|>"})
            return True
        except ValueError:
            return False

    def get_random_vocab_item(self):
        index = random.randrange(self.get_vocab_size())
        return self.decode([index])


    def get_eot_token(self):
        enc = GPT2_ENC # tiktoken.get_encoding("gpt2")
        return enc.eot_token



    def encode(self, text): # text can be a string to be encoded or an int array already encoded
        if isinstance(text, str):
            enc = GPT2_ENC # tiktoken.get_encoding("gpt2")
            return enc.encode(text, allowed_special={"<|endoftext|>"})
        else:
            return text # already encoded int array
            

    # can return incorrect utf-8 sequences, chars will be replaced with ? chars
    def decode(self, ids, errors='replace'):
        ''' Incorrect utf-8 code points are dealt with errors param of bytes.decode()

            ids can be: an int or list of ints (str output), or a numpy 2D array (str list output) "
        '''

        enc = GPT2_ENC # tiktoken.get_encoding("gpt2")

        if isinstance(ids,np.ndarray):
            assert ids.ndim == 2, "Only 2d numpy arrays supported"
            str_out = False

        elif isinstance(ids,int):
            ids = np.array([[ids]]) # shape=(1,t)
            str_out = True

        elif isinstance(ids, list):
            ids = np.array([ids]) # shape=(1,t)
            str_out = True

        else:
            assert False, "Only numpy 2D arrays, int lists or int supported"

        #print(ids.shape, ids)
        b,t=ids.shape
        out=[]

        for ib in range(b):
            row = ids[ib,:].tolist()
            text = enc.decode(row, errors=errors)
            out.append(text)

        if str_out:
            return out[0] # type str
        else:
            return out # type str list with len = b


    def bufd_decode(self, ids):
        ''' Buffered token bytes -> utf-8 decoding. If tokens don't map to a valid utf-8 char, we buffer the bytes and prepend it to next call's, which will hopefully allow decoding to utf-8

            ids can be: an int or list of ints (str output), or a numpy 2D array (str list output) "
        '''

        enc = GPT2_ENC # tiktoken.get_encoding("gpt2")

        if isinstance(ids,np.ndarray):
            assert ids.ndim == 2, "Only 2d numpy arrays supported"
            str_out = False

        elif isinstance(ids,int):
            ids = np.array([[ids]]) # shape=(1,t)
            str_out = True

        elif isinstance(ids, list):
            ids = np.array([ids]) # shape=(1,t)
            str_out = True

        else:
            assert False, "Only numpy 2D arrays, int lists or int supported"

        #print(ids.shape, ids)
        b,t=ids.shape
        out=[]

        if len(self.bufd_decode_list) != b: # resize and clear
            self.bufd_decode_list = [b''] * b

        for ib in range(b):
            row = ids[ib,:].tolist()

            if len(row):
                dec_list = enc.decode_tokens_bytes(row) # a list with bytes
            else: # for flush
                dec_list = []

            bl=self.bufd_decode_list[ib]
            for b in dec_list:            
                bl = bl + b

            text,undec = consumme_decode_utf8(bl)
            self.bufd_decode_list[ib] = undec
            out.append(text)

        if str_out:
            return out[0] # type str
        else:
            return out # type str list with len = b



    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]

        # return as tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y






# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    UTF-8 character dataset. index <=> full utf-8 character
    """


    @staticmethod
    def load_train_val_datasets(train_path, val_path_or_train_split,
                                block_size, 
                                repeat_if_needed=False, 
                                verbose=True):
        """ returns train_dataset, val_dataset - val dataset can be None """

        data = CharDataset.load_data(data_path=train_path, verbose=verbose)

        if isinstance(val_path_or_train_split, str): # val from path

            val_data = CharDataset.load_data(data_path=val_path_or_train_split, verbose=verbose)

            # calc combined vocab
            vocab_chars = CharDataset.calc_vocab_chars(data + val_data)

            train = CharDataset(block_size, data=data, repeat_if_needed=repeat_if_needed, vocab_chars=vocab_chars, verbose=verbose)
            val = CharDataset(block_size, data=val_data, repeat_if_needed=repeat_if_needed, vocab_chars=vocab_chars, verbose=verbose)

        else: # split from train
            assert isinstance(val_path_or_train_split, float), "val_path_or_train_split can be of str (path) or float (train_split) types"

            assert val_path_or_train_split > 0. and val_path_or_train_split <= 1., "0 < train split <= 1"

            split_index = int(len(data) * val_path_or_train_split)

            train = CharDataset(block_size, data=data[:split_index], repeat_if_needed=repeat_if_needed, verbose=verbose)

            if split_index < len(data):
                vocab_chars = train.get_vocab_chars()            
                val = CharDataset(block_size, data=data[split_index:], repeat_if_needed=repeat_if_needed,
                                  vocab_chars=vocab_chars, verbose=verbose)
            else:
                val = None

        return train, val



    @staticmethod
    def load_data(data=None, data_path=None, verbose=True):

        assert (data is not None) ^ (data_path is not None), "Dataset does not support dummy data: one of data and data_path must be given"

        if data_path is not None:
            with open(data_path, 'r', newline=None) as f: # newlines converted to \n
                data = f.read()

        if data is None: # dummy dataset with size 1 and first token repeated
            assert False, "does not support dummy dataset"

        return data


    @staticmethod
    def calc_vocab_chars(data):
        return sorted( list(set(data)) )


    def __init__(self, block_size, data=None, data_path=None, repeat_if_needed=False, 
                 vocab_chars=None,
                 verbose=True):

        """ if vocab_chars is passed, it will be used instead of data's """

        assert (data is not None) ^ (data_path is not None), "only one of data and data_path must be given - does not support dummy data"


        data = CharDataset.load_data(data=data, data_path=data_path, verbose=verbose)
        # self.data is now encoded as uint16 tokens

        if vocab_chars is not None:            
            chars = sorted( vocab_chars ) # sorted just in case
        else:
            chars = CharDataset.calc_vocab_chars(data)

        self.vocab_size = len(chars)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        if len(data) < block_size+1 and repeat_if_needed:
            times = int(math.ceil((block_size+1) / len(data)))
            if verbose:
                print(f"Expanding initial dataset size of {len(data)} (less than block_size+1) by {times} times to size of {times*len(data)}")
            data = data * times

        assert len(data) > block_size, f"data length ({len(data)}) must be greater than block_size({block_size})"

        self.data = data
        self.data_len = len(data) - block_size

        self.block_size = block_size



    def get_vocab_chars(self):
        return list(self.stoi.keys())

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return self.data_len


    def is_text_valid(self, text):
        return all([t in self.stoi for t in text])

    def get_random_vocab_item(self):
        index = random.randrange(self.get_vocab_size())
        return self.itos[index]

    def get_eot_token(self):
        return None


    def encode(self, text):
        return [self.stoi[id] for id in text]


    def decode(self, ids):        
        return ''.join([self.itos[int(i)] for i in ids])

    def bufd_decode(self, ids):
        """ chars are always utf-8 complete, so just use normal decode """
        return self.decode(ids)


    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = self.encode(chunk)
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y




# -----------------------------------------------------------------------------

DATASET_CLASS_MAP = {'gpt2': GPT2TokensDataset, 'char': CharDataset}

def dataset_class_from_name(class_name):
    return DATASET_CLASS_MAP[class_name]


