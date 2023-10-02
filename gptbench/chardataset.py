"""
Char-based dataset classes
"""

import os, sys, copy, array, random, math, pdb

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

from .utils import is_utf8, consumme_decode_utf8, bool_from_any





# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    UTF-8 character dataset. index <=> full utf-8 character
    """


    @staticmethod
    def load_train_val_datasets(block_size, 
                                train_path, train_split=None, val_path=None,
                                verbose=True,

                                repeat_if_needed=False, 
                                **ignore_other_kwargs):

        """ returns train_dataset, val_dataset - val dataset can be None """

        # extra params (after verbose) can come from strings, convert eventual non-str:
        repeat_if_needed = bool_from_any(repeat_if_needed)

        data = CharDataset.load_data(data_path=train_path, verbose=verbose)

        val = None
        if val_path is not None: # val from path

            val_data = CharDataset.load_data(data_path=val_path, verbose=verbose)

            # calc combined vocab
            shared_vocab_chars = CharDataset.calc_vocab_chars(data + val_data)

            train = CharDataset(block_size, data=data, 
                                repeat_if_needed=repeat_if_needed, 
                                shared_vocab_chars=shared_vocab_chars, verbose=verbose)
            val = CharDataset(block_size, data=val_data, 
                              repeat_if_needed=repeat_if_needed, 
                              shared_vocab_chars=shared_vocab_chars, verbose=verbose)

        elif train_split is not None:

            assert train_split > 0. and train_split <= 1., "0 < train split <= 1"

            split_index = int(len(data) * train_split)

            train = CharDataset(block_size, data=data[:split_index], 
                                repeat_if_needed=repeat_if_needed, verbose=verbose)

            if split_index < len(data):
                shared_vocab_chars = train.get_vocab_items()            
                val = CharDataset(block_size, data=data[split_index:], 
                                  repeat_if_needed=repeat_if_needed,
                                  shared_vocab_chars=shared_vocab_chars, verbose=verbose)

        else:
            train = CharDataset(block_size, data=data, 
                                repeat_if_needed=repeat_if_needed, verbose=verbose)

        return train, val



    @staticmethod
    def load_data(data=None, data_path=None, verbose=True):

        assert (data is not None) ^ (data_path is not None), "Dataset does not support dummy data: one of data or data_path must be given"

        if data_path is not None:
            with open(data_path, 'r', encoding='utf-8', newline=None) as f: # newlines converted to \n
                data = f.read()

        if data is None: # dummy dataset with size 1 and first token repeated
            assert False, "does not support dummy dataset"

        return data


    @staticmethod
    def calc_vocab_chars(data):
        return sorted( list(set(data)) )


    def __init__(self, block_size, data=None, data_path=None, repeat_if_needed=False, 
                 shared_vocab_chars=None,
                 verbose=True):

        """ if shared_vocab_chars is passed, it will be used instead of data's """

        assert (data is not None) ^ (data_path is not None), "only one of data and data_path must be given - does not support dummy data"


        data = self.load_data(data=data, data_path=data_path, verbose=verbose)

        if shared_vocab_chars is not None:            
            chars = shared_vocab_chars
        else:
            chars = self.calc_vocab_chars(data)

        self.vocab_size = len(chars)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        if len(data) < block_size+1 and repeat_if_needed:
            times = int(math.ceil((block_size+1) / len(data)))
            if verbose:
                print(f"Expanding initial dataset size of {len(data)} (less than block_size+1) by {times} times to size of {times*len(data)}")
            data = data * times


        self.data = data
        self.data_len = len(data) - block_size

        self.block_size = block_size


    def get_src_data(self):
        return self.data

    def get_src_sample(self, index):
        return self.data[index:index + self.block_size]

    def get_data(self): 
        return self.data


    def get_vocab_items(self):
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

        if isinstance(ids,np.ndarray):
            assert ids.ndim == 2, "numpy array param: only 2d arrays supported"
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
            text = ''.join([self.itos[int(i)] for i in row])
            out.append(text)

        if str_out:
            return out[0] # type str
        else:
            return out # type str list with len = b


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

class PaddedLineCharDataset(Dataset):
    """
    UTF-8 character dataset. index <=> full utf-8 character. Read from line-based text files.
    Each sample is padded at the right with a given char. Last X char predicts a -1 index in Y, no used for loss calc.
    Empty lines will be removed.
    """


    @staticmethod
    def load_train_val_datasets(block_size, 
                                train_path, train_split=None, val_path=None,
                                verbose=True,

                                line_sep_char='\n',
                                pad_char='\0',
                                shuffle=False,
                                **ignore_other_kwargs):
        """ returns train_dataset, val_dataset - val dataset can be None """

        # extra params (after verbose) can come from strings, convert eventual non-str:
        shuffle = bool_from_any(shuffle)

        data = PaddedLineCharDataset.load_data(data_path=train_path, verbose=verbose)

        val = None

        if val_path is not None: # val from path

            val_data = PaddedLineCharDataset.load_data(data_path=val_path, verbose=verbose)

            # calc combined vocab
            shared_vocab_chars = PaddedLineCharDataset.calc_vocab_chars(data + val_data, 
                                                                        line_sep_char=line_sep_char,
                                                                        pad_char=pad_char)

            train = PaddedLineCharDataset(block_size, data=data, 
                                          line_sep_char=line_sep_char, pad_char=pad_char, shuffle=shuffle,
                                          shared_vocab_chars=shared_vocab_chars, verbose=verbose)

            val = PaddedLineCharDataset(block_size, data=val_data,
                              line_sep_char=line_sep_char, pad_char=pad_char, shuffle=shuffle,
                              shared_vocab_chars=shared_vocab_chars, verbose=verbose)

        elif train_split is not None: # split from train

            assert train_split > 0. and train_split <= 1., "0 < train split <= 1"

            split_index = PaddedLineCharDataset.line_start_char_from_ratio(data, line_sep_char=line_sep_char, line_ratio=train_split)

            train = PaddedLineCharDataset(block_size, data=data[:split_index],
                                line_sep_char=line_sep_char, pad_char=pad_char, shuffle=shuffle,
                                verbose=verbose)

            if split_index < len(data):
                shared_vocab_chars = train.get_vocab_items()            
                val = PaddedLineCharDataset(block_size, data=data[split_index:],
                                  line_sep_char=line_sep_char, pad_char=pad_char, shuffle=shuffle,
                                  shared_vocab_chars=shared_vocab_chars, verbose=verbose)

        else:
            train = PaddedLineCharDataset(block_size, data=data,
                                line_sep_char=line_sep_char, pad_char=pad_char, shuffle=shuffle,
                                verbose=verbose)

        return train, val



    @staticmethod
    def load_data(data=None, data_path=None, verbose=True):

        assert (data is not None) ^ (data_path is not None), "Dataset does not support dummy data: one of data or data_path must be given"

        if data_path is not None:
            with open(data_path, 'r', encoding='utf-8', newline=None) as f: # newlines converted to \n
                data = f.read()

        if data is None: # dummy dataset with size 1 and first token repeated
            assert False, "does not support dummy dataset"

        return data


    @staticmethod
    def calc_vocab_chars(data, line_sep_char, pad_char):
        """ Includes padding char, but not line_sep """
        chars = list(set(data + pad_char))

        if line_sep_char in chars:
            chars.remove(line_sep_char)

        return sorted(chars)

    @staticmethod
    def line_start_char_from_ratio(data, line_sep_char, line_ratio):
        """ Return the char position ratio that matches the beginning of a line"""

        total_chars = len(data)
        if line_ratio >= 1.:
            return total_chars

        total_lines = data.count(line_sep_char) + 1
        target_line_index = int(line_ratio * total_lines)
        char_pos=0
        l=0
        while True:
            if l == target_line_index:
                break

            pos = data.find('\n', char_pos)
            if pos == -1:
                break

            char_pos = pos+1
            l+=1

        return char_pos



    def __init__(self, block_size, data=None, data_path=None,
                 line_sep_char='\n',
                 pad_char=None,
                 shuffle=False,
                 shared_vocab_chars=None,
                 verbose=True):

        """
        If shared_vocab_chars is passed, it will be used instead of data's. shared_vocab_chars with or without pad_char included at index 0.
        """

        assert (data is not None) ^ (data_path is not None), "only one of data and data_path must be given - does not support dummy data"

        assert pad_char is not None, "pad_char must be given"


        data = self.load_data(data=data, data_path=data_path, verbose=verbose)
        self.src_data = data

        if shared_vocab_chars is not None:            
            chars = shared_vocab_chars
        else:
            chars = PaddedLineCharDataset.calc_vocab_chars(data, line_sep_char=line_sep_char, pad_char=pad_char)

        self.vocab_size = len(chars)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data = data.split(line_sep_char)

        data = [d for d in data if len(d)] # remove empty entries

        assert len(data), "Dataset cannot be empty empty"

        if bool_from_any(shuffle):
            random.shuffle(data)

        max_len = max([len(line) for line in data])

        assert max_len <= block_size, f"Dataset includes lines with more characters ({max_len}) than block_size={block_size}"

        self.data = data
        self.data_len = len(data)

        self.block_size = block_size


    def get_src_data(self):
        return self.src_data

    def sample_split(self, start_index, stop_index,
                     sep, sep_included):

        """ Split samples by sep char into two lists a and b. sep_included: -1 at end of a, 1 at beginning of b, 0 not included """

        alist=[]; blist=[]

        for index in range(start_index, stop_index):
            s = self.data[index]
            l = s.split(sep, maxsplit=1)            
            assert len(l) == 2, f"Unable to split: sep char not found in dataset entry '{s}'"

            a,b = l
            if sep_included:
                if sep_included==-1:
                    a += sep
                else:
                    b = sep + b

            alist.append(a)
            blist.append(b)

        return alist,blist







    def get_data(self): 
        return self.data


    def get_vocab_items(self):
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

        if isinstance(ids,np.ndarray):
            assert ids.ndim == 2, "numpy array param: only 2d arrays supported"
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

            try:
                index = row.index(0) # pad char is 0
                row=row[:index]
            except ValueError:
                ... # full row

            text = ''.join([self.itos[int(i)] for i in row])
            out.append(text)

        if str_out:
            return out[0] # type str
        else:
            return out # type str list with len = b


    def bufd_decode(self, ids):
        """ chars are always utf-8 complete, so just use normal decode """
        return self.decode(ids)



    def __getitem__(self, idx):

        line = self.data[idx]
        # encode every character to an integer
        dix = self.encode(line)

        # zero is the pad index
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.zeros(self.block_size, dtype=torch.long)

        ll = len(dix)
        x[:ll] = torch.as_tensor(dix)
        y[:ll-1] = torch.as_tensor(dix[1:])
        # would never stop! y[ll-1] = -1 # prediction for last char in x shall not be used for loss calc

        """
        n_zerozero = 3 # how many 0->0 to leave
        if ll+n_zerozero < self.block_size: # mark all after last y==0 to be -1 to not be accounted for loss
            y[ll+n_zerozero:] = -1
        """

        return x, y







