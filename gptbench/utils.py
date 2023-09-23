import os, sys, random, json, gc

import torch
from torch.nn import functional as F

import numpy as np

# -----------------------------------------------------------------------------

def set_seed(seed, verbose=True):
    if seed == 0:
        random.seed()
        seed = random.randrange(2**32)

    if verbose:
        print(f"New random seed {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def in_notebook():
    """ https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False

    return True


def die(msg=None, exit_code=1):

    if in_notebook():
        ex = f'exit code: {exit_code}'
        if msg is not None:
            assert False, msg + ' - ' + ex
        else:
            assert False, ex
    else:
        if msg is not None:
            print(msg)
        sys.exit(exit_code)





# top_p or nucleus sampling: cuts values with probability mass > p. 0.1. 
# example: 0.1=only top 10% probable values, 0.9=90% of the most probable values
# adapted from Hugging Face transformers' top_k_top_p_filtering()
# https://github.com/huggingface/transformers
# https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_p(logits, top_p, min_tokens_to_keep=1):

    assert top_p < 1.0, "0 < top_p < 1"

    # sort descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # softmax then calc cummulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # remove entries with cumulative probability above the threshold (entries with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > top_p

    if min_tokens_to_keep > 1: # keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    # shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float("inf")

    return logits






class CfgNode:
    """ a lightweight configuration class inspired by yacs """


    @staticmethod
    def from_sysargv(sys_argv, key_must_exist):

        argv = sys_argv[1:]

        c = empty_config()
        c.merge_from_args(argv, key_must_exist=key_must_exist)

        return c


    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper()

    def _str_helper(self):
        """ need to have a helper to support nested indentation for pretty printing """
        own = []
        sub = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                if len(sub):
                    sub.append("\n")
                sub.append("%s: " % k)
                sub.append(v._str_helper())
            else:
                own.append("%s=%s " % (k, v))
                #own.append("%s: %s (%s) " % (k, v, type(v)))

        if len(sub):
            own += ["\n"]
        out = "".join(own + sub)

        return out # : for root vars

    def to_dict(self, include_non_jsonable, only_root_keys=None):
        """ return a dict representation of the config """
        out = {}

        for k, v in self.__dict__.items():

            if only_root_keys is not None:
                if k not in only_root_keys:
                    continue

            if isinstance(v, CfgNode):
                out[k] = v.to_dict(include_non_jsonable)
            elif isinstance(v, dict) or isinstance(v, list) or isinstance(v, str) or isinstance(v, float) or isinstance(v, int) or isinstance(v, bool) or v is None:
                out[k] = v
            elif include_non_jsonable: # ignore others like dataset class that can't be converted to json
                out[k] = v

        return out




    @staticmethod
    def _typed_from_any(a):
        ''' returns type None, int, float, str values from whatever-typed input '''

        def is_number(val):
            try:
                float(val)
                return True
            except ValueError:
                return False

        s = str(a)
        if s == 'None':
            return None
        elif is_number(s):
            f = float(s)
            if f % 1. == 0.:
                return int(f)
            else:
                return f
        else:
            return str(s)


    def _obj_key_from_name(self, name):
        keys = name.split('.')
        obj = self
        for k in keys[:-1]:
            if not hasattr(obj, k):
                return None,None
            obj = getattr(obj, k)

        leaf_name = keys[-1]
        if hasattr(obj, leaf_name):
            return obj,leaf_name
        else:
            return obj,None


    def has(self, name):
        obj,leaf = self._obj_key_from_name(name)
        return leaf is not None

    def get_or(self, name, default_value):
        obj,leaf = self._obj_key_from_name(name)
        if leaf is not None:
            return getattr(obj, leaf)
        else:
            return default_value

    def _set(self, name,value):

        keys = name.split('.')
        obj = self
        for k in keys[:-1]:
            if not hasattr(obj, k):
                return False
            obj = getattr(obj, k)

        leaf_name = keys[-1]
        setattr(obj, leaf_name, value)
        return True

    def set(self, *args, **kwargs):

        if len(args) == 2:
            return self._set(*args)
        else: # local name=value
            self.__dict__.update(kwargs)        
            return True # can't fail

    def set_if_unset(self, name, value):
        """ 1: set, 0: alreay set, -1: unable to set (bad path) """
        keys = name.split('.')
        obj = self
        for k in keys[:-1]:
            if not hasattr(obj, k):
                return -1
            obj = getattr(obj, k)

        leaf_name = keys[-1]
        if hasattr(obj, leaf_name):
            return 0
        else:
            setattr(obj, leaf_name, value)
            return 1


    def merge_from_dict(self, d, only_root_keys=None, existing_only=False):
        """ only_root_keys: local-level key names that can be merged - others are ignored """

        if only_root_keys is not None:
            d2 = {}
            for key,val in d.items():
                if key in only_root_keys:
                    d2[key] = val
            d = d2

        for k,v in d.items():
            if not existing_only or self.has(k):
                self._set(k,v)



    def merge_from_config(self, other_config, only_root_keys=None):
        """ copies all keys from other_config, then recurses into existing CfgNodes that both have """

        for k, v in other_config.__dict__.items():

            if only_root_keys is not None:
                if k not in only_root_keys:
                    continue

            if isinstance(v, CfgNode) and hasattr(self,k) and isinstance(getattr(self,k), CfgNode):
                getattr(self,k).merge_from_config(v)
            else:
                setattr(self, k, v)

    def merge_from_args(self, args, key_must_exist):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `-arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        -seed=1117 -work_dir=out -model.n_layer=10 -trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            l = len(keyval)
            if l < 2: # -resume -> -resume=1
                keyval.append(1)
            elif l > 2: # take care of accepting multiple '=' like -start="1+1=?"
                keyval[1] = '='.join(keyval[1:])
                keyval = keyval[:2]

            key, val = keyval # unpack

            val = CfgNode._typed_from_any(val)

            # find the appropriate object to insert the attribute into
            if key[:1] == '-': key = key[1:] # strip eventual first '-'
            if key[:1] == '-': key = key[1:] # strip eventual second '-'

            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            if key_must_exist:
                assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            #print("Config overwriting %s=%s" % (key, val))
            setattr(obj, leaf_key, val)

        




def is_utf8(buffer_or_str):
    if isinstance(buffer_or_str, list):
        return False
    elif isinstance(buffer_or_str, str):
        try:
            buffer_or_str.encode(encoding='utf-8', errors='strict')            
            return True
        except UnicodeError:
            return False
    else:
        try:
            buffer_or_str.decode('utf-8', errors='strict')
            return True
        except UnicodeError:
            return False



def consumme_decode_utf8(b):
    """
    Decode as many utf-8 code points as possible, returning those and the unconsummed bytes.
    Param b is a bytes variable.
    See https://en.wikipedia.org/wiki/UTF-8
    """

    text=''

    try:
        while lb :=len(b):

            if b[0] <= 0x007f:
                text += chr(b[0])
                b = b[1:]

            elif lb >= 2 and ( 
                 ((b[0]>>5) & 0b111 == 0b110) and ((b[1]>>6) & 0b11 == 0b10)
                 ):
                text += b[:2].decode('utf-8', errors='strict')
                b = b[2:]

            elif lb >= 3 and (
                 ((b[0]>>4) & 0b1111 == 0b1110) and ((b[1]>>6) & 0b11 == 0b10) and ((b[2]>>6) & 0b11 == 0b10)
                 ):
                text += b[:3].decode('utf-8', errors='strict')
                b = b[3:]

            elif lb >= 4 and (
                 ((b[0]>>3) & 0b11111 == 0b11110) and ((b[1]>>6) & 0b11 == 0b10) and ((b[2]>>6) & 0b11 == 0b10) and ((b[3]>>6) & 0b11 == 0b10)
                 ):
                text += b[:4].decode('utf-8', errors='strict')
                b = b[4:]
            else:
                break

    except UnicodeDecodeError:
        ...

    return text,b



def print_sepline():
    print('-' * 80)





def cuda_max_memory_init():
    if not torch.cuda.is_available(): return

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def cuda_max_memory():
    if not torch.cuda.is_available(): return

    torch.cuda.synchronize()
    b = torch.cuda.max_memory_allocated()
    cuda_max_memory_init()


    return f"CUDA max memory used: {b/1e6:.2f}M"
