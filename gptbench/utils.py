import os, sys, random, json, gc

import torch
from torch.nn import functional as F


# -----------------------------------------------------------------------------

def set_all_random_seeds(seed):
    if seed == 0:
        random.seed()
        seed = random.randrange(2**32)

    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed



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



def str_dict_from_str(st):
    """
    Convert a string in the form name1=value1,name2=value2,... into a dict.
    Escape commas in values as \,. Values can contain '=' without problems
    """
    
    st = st.replace('\,', '\0')

    out = {}

    pairs = st.split(',')
    for kv in pairs:

        keyval = kv.split('=', maxsplit=1)
        assert len(keyval) == 2, "Must be in the form name=value"

        key, val = keyval # unpack
        val = val.replace('\0','\,')

        out[key] = val

    return out




def bool_from_any(v):
    if isinstance(v,bool):
        return v
    elif v is None:
        return False

    def is_number(val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    if is_number(v):
        return abs(v) >= sys.float_info.epsilon
    else: # assume str
        v = str(v)
        return v != '' and v != '0' and v != 'False'






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
