"""
Dataset classes derived from torch.utils.data.Dataset
"""

import os, sys, copy, array

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import tiktoken

from gptbench.utils import CfgNode, is_utf8


class DatasetBase(Dataset):

    @staticmethod
    def get_default_config():
        c = CfgNode()

        c.cls = None
        c.path = None # default '' means dummy dataset with one sample, for sampling (for token encode/decode)
        c.train = None
        c.val = None
        c.trainsplit = 0.9

        return c

    @staticmethod
    def checkpoint_config_keys():
        return ["path", "trainsplit"]


    @staticmethod
    def create_train_val_datasets(block_size, cls, trainsplit, data=None, data_path=None):
        """ returns train_dataset, val_dataset - val dataset can be None if trainsplit=1. """

        assert trainsplit is not None and trainsplit > 0. and trainsplit <= 1., "0 < trainsplit <= 1"

        data = cls.load_data(block_size, data=data, data_path=data_path, verbose=True)

        # deal with dummy dataset: return copies of data
        if data is None and data_path is None:
            return data, copy.copy(data) if trainsplit < 1. else None

        split_index = int(len(data) * trainsplit)

        train = cls(block_size, data=data[:split_index])

        if split_index < len(data):
            val = cls(block_size, data=data[split_index:])
        else:
            val = None

        return train, val




''' 
Don't store tiktoken in object, because tokenizer is not pickable: https://github.com/huggingface/datasets/issues/5769

DataLoader with num_workers > 0 will error:
TypeError: cannot pickle 'builtins.CoreBPE' object
'''
GPT2_ENC = tiktoken.get_encoding("gpt2")


class GPT2TokensDataset(DatasetBase):
    """
    Emits sequences of block_size GPT2 tokens randomly sampled from a dataset
    """

    @staticmethod
    def load_data(block_size, data=None, data_path=None, verbose=False):

        enc = GPT2_ENC # tiktoken.get_encoding("gpt2")

        if data_path is not None:
            with open(data_path, 'rb') as f:
                data = f.read()

        if data is None: # dummy dataset with size 1 and first token repeated
            if verbose:
                print("Dataset: dummy 0 tokens")
            data = [0] * (block_size+1)

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


    def __init__(self, block_size, data=None, data_path=None):
        """ data is a np.array(dtype=np.uint16) 
        if both data and data_path are None, a dummy dataset is created with len=1 """

        self.data = GPT2TokensDataset.load_data(block_size, data=data, data_path=data_path)
        # self.data is now encoded as uint16 tokens

        assert len(self.data) > block_size, f"data length ({len(self.data)}) must be greater than block_size({block_size})"

        enc = GPT2_ENC # tiktoken.get_encoding("gpt2")

        self.vocab_size = enc.n_vocab
        self.block_size = block_size

        self.data_len = len(self.data) - self.block_size

        self.decode_buffer = b'' # partial utf-8 undecoded byte list


    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def get_eot_token(self):
        enc = GPT2_ENC # tiktoken.get_encoding("gpt2")
        return enc.eot_token

    def __len__(self):
        return self.data_len

    def encode(self, text): # text can be a string to be encoded or an int array already encoded
        if isinstance(text, str):
            enc = GPT2_ENC # tiktoken.get_encoding("gpt2")
            return enc.encode(text, allowed_special={"<|endoftext|>"})
        else:
            return text # already encoded int array

    # can return incorrect utf-8 sequences, chars will be replaced with ? chars
    def decode(self, ids):
        enc = GPT2_ENC # tiktoken.get_encoding("gpt2")
        text = enc.decode(ids, errors='replace')
        return text


    def bufd_decode(self, ids):
        ''' buffered token bytes -> utf-8 decoding. If token(s) doesn't map to a valid utf-8 char we buffer the bytes and prepend it to next call's, which will hopefully allow decoding to utf-8 '''
        if len(ids):
            enc = GPT2_ENC # tiktoken.get_encoding("gpt2")
            blist = enc.decode_tokens_bytes(ids)
        else: # for flush
            blist = []

        bl=self.decode_buffer
        for b in blist:            
            bl = bl + b

        try:
            text = bl.decode("utf-8", errors='strict')
            self.decode_buffer = b''
            return text
        except UnicodeDecodeError:
            self.decode_buffer = bl # accumulate. TODO: We could try decoding all bytes up to an invalid one
            return ''

    def bufd_flush(self):
        text = self.bufd_decode([])
        self.decode_buffer = b'' # ensure buffer cleaned
        return text


    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]

        # return as tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y







class CharDataset(Dataset):
    """
    Emits sequences of block_size characters randomly sampled from a dataset
    """

    def __init__(self, block_size, data=None, data_path=None):

        assert data is not None or data_path is not None

        if data_path is not None and data_path != '':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = f.read()

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size



        self.data = data
        self.data_len = data_size - self.block_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def get_eot_token(self):
        return None

    def __len__(self):
        return self.data_len

    def encode(self, text):
        return [self.stoi[id] for id in text]

    def decode(self, ids):
        return ''.join([self.itos[int(i)] for i in ids])


    def bufd_decode(self, ids):
        return self.decode(ids)

    def bufd_flush(self):
        return ''


    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = self.encode(chunk)
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y



