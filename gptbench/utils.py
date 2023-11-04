import os, sys, random, json, gc
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F


# -----------------------------------------------------------------------------

def set_all_random_seeds(seed):
    if seed == 0:
        random.seed()
        seed = random.randrange(2**32)
    else:
        seed = seed & (2**32-1)

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





# adapted from Hugging Face transformers' top_k_top_p_filtering()
# https://github.com/huggingface/transformers
# https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_p(logits, top_p, min_tokens_to_keep=1):
    """
    Top_p or nucleus sampling: only top most probable tokens with accumulated probabilities up to top_p are kept for sampling. 
    At least min_tokens_to_keep are preserved.

    Example:
    0.4: (more picky) only 40% top probabilities (mass) are kept (or min_tokens_to_keep)
    0.9: (less picky) all of the top 90% accumulated probabilities are kept.
    """

    assert top_p > 0. and top_p < 1.0, "0 < top_p < 1"

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




def plot_loss_chart(loss_arr, dest=None, title=None, dpi=200):
    """
    loss_arr: iter_num,train_loss or iter_num,train_loss,val_loss
    dest: None to plt.show() or path to image file for saving
    """

    iters = [int(e[0]) for e in loss_arr]
    train_loss = [float(e[1]) for e in loss_arr]
    val_loss = [float(e[2]) for e in loss_arr] if len(loss_arr[0]) > 2 else None


    plt.figure(dpi=dpi)
    if title is not None:
        plt.title(title)
        
    plt.plot(iters,train_loss, label='train', linewidth=1, color='blue', marker=".")
    if val_loss:
        plt.plot(iters,val_loss, label='validation', linewidth=1, color='green', marker=".") # , linestyle='--'

    plt.xlabel("Iterations")
    # plt.xticks(iters)
    plt.ylabel("Loss")

    plt.legend(loc="upper right")
    #plt.ylim(0.)

    if dest is not None:
        plt.savefig(dest)
        plt.close()
    else:
        plt.show()




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





def cuda_memory_stats():
    """
    See https://discuss.pytorch.org/t/free-all-gpu-memory-used-in-between-runs/168202
    """
    if not torch.cuda.is_available(): return

    print(f"cuda mem: allocated={torch.cuda.memory_allocated()/1024**2:.1f}MB, cached={torch.cuda.memory_reserved()/1024**2:.1f}MB")

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
