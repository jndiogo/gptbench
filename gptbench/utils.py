import os, sys, random, json

import torch
from torch.nn import functional as F

import numpy as np

# -----------------------------------------------------------------------------

def set_seed(seed):
    if seed == 0:
        random.seed()
        seed = random.randrange(2**32)

    print(f"New random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def last_config_save(config):

    d = config.to_dict(False)

    work_dir = config.work_dir

    with open(os.path.join(work_dir, 'last_config.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(d, indent=4))




def die(msg, exit_code=1):
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

    def merge_from_dict(self, d, only_keys=None):

        if only_keys is not None:
            d2 = {}
            for key,val in d.items():
                if key in only_keys:
                    d2[key] = val
            d = d2

        self.__dict__.update(d)


    def merge_from_config(self, other_config):
        """ copies all keys from other_config, then recurses into existing CfgNodes that both have """

        for k, v in other_config.__dict__.items():

            if isinstance(v, CfgNode) and hasattr(self,k) and isinstance(getattr(self,k), CfgNode):
                getattr(self,k).merge_from_config(v)
            else:
                setattr(self, k, v)


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



def print_sepline():
    print('-' * 80)