"""

"""

import os, sys, copy, signal, json
from enum import IntFlag

import torch


from .model import GPT
from .trainer import Trainer

from .utils import CfgNode



# -----------------------------------------------------------------------------
"""
How config works:

1) User code sets values on a config object got from empty_config()

2) Any command line args are merged into that config with merge_config_from_sysargv()

3) This partial config is then passed into run(). It's merged as needed into a default full config and any resumed checkpoint information, to get the final fully resolved config


"""


def empty_config():
    """ an empty config: options that are later set will override the full config """
    c = CfgNode()

    # train
    c.train = CfgNode()

    # sample
    c.sample = CfgNode()


    # dataset
    c.dataset = CfgNode()

    # model
    c.model = CfgNode()

    # trainer
    c.trainer = CfgNode()

    return c


def default_full_config():
    """ returns a full config with all possible values """

    c = empty_config()

    c.seed = 0 # 0 means random seed

    # train
    c.train = train_get_default_config()

    # sample
    c.sample = sample_get_default_config()


    # dataset
    c.dataset = dataset_get_default_config()

    # model
    c.model = GPT.get_default_config()

    # trainer
    c.trainer = Trainer.get_default_config()

    return c





class LogFlag(IntFlag):
    NONE = 0

    INIT = 1

    BATCH_DOT = 4
    BATCH_LOSS = 8
    EVAL_LOG = 16
    TRAIN = BATCH_DOT | BATCH_LOSS | EVAL_LOG

    CUDA_MEMORY = 256

    ALL = INIT | BATCH_DOT | BATCH_LOSS | EVAL_LOG | CUDA_MEMORY





# -----------------------------------------------------------------------------
def sample_get_default_config():

    # sample.*
    c = CfgNode()

    c.count = 1 # number of generations
    c.len = 100 # token count
    
    c.start = None # None: use random vocabulary item on each sampling. Or str with starting text
    c.start_after = None
    c.stop_before = None

    c.pertoken = 1 # display each token immediately
    c.eotstop = 0 # 0 don't stop, -1 stop before, 1 stop after (and display it)

    c.top = 0 # top_k/top_p  0: off,  ]0..1]: top_p,  [-1..0[: top_k(vocab_size * -top),  >=1: top_k(int(n))
    c.temp = 1. # temperature

    c.multiline = 0 # prompt mode: input multiple lines until a Ctrl+D or Ctrl+Z (in Windows)

    return c

def sample_checkpoint_config_keys():
    return ['count', 'len', 'start', 'pertoken', 'eotstop', 'top', 'temp', 'multiline']






# -----------------------------------------------------------------------------
def train_get_default_config():

    # train.*
    c = CfgNode()

    c.start_iter_num = 0
    c.start_eval_loss = float('inf')
    c.start_train_loss = float('inf')
    c.start_val_loss = float('inf')

    c.log_period = -0.1 # simple forward pass loss log. Negative numbers mean max(1, int(eval_period * -log_period))

    c.eval_period = 100 # each n batches we eval and check if saving model. 0 for none
    c.eval_type = 2 # how to estimate loss -> 1: on test data, 2: on val data (or test if no val dataset), 1|2=3: mean(test,val)
    c.eval_iters = 100
    c.eval_save_checkpt = 1 # 0=never, 1=on lower loss, 2=always

    c.sample_period = 1000 # when to sample. 0 for never

    c.debug = 0 # 0: none, 1: log cuda peak used memory on each evaluation, 2: print '.' per batch

    return c


def train_checkpoint_config_keys():
    return ['start_iter_num', 'start_eval_loss', 'start_train_loss', 'start_val_loss', 'eval_period', 'eval_type', 'eval_iters', 'sample_period']




# -----------------------------------------------------------------------------
def dataset_get_default_config():
    c = CfgNode()

    c.class_name = None
    c.train_path = None
    c.val_path_or_train_split = 0.9 # 0..1 float: train_split for validation dataset from train dataset, str: validation dataset path

    return c

def dataset_checkpoint_config_keys():
    return ['class_name', 'train_path', 'val_path_or_train_split']







# -----------------------------------------------------------------------------
def merge_config_from_sysargv(sys_argv, base_config = None):

    argv = sys_argv[1:]

    if len(argv) < 2: # at least mode, init
        usage()
        die()

    if base_config is not None:
        config = base_config
    else:
        config = empty_config()

    config.merge_from_args(argv, key_must_exist=False)

    if not (hasattr(config, 'mode') and hasattr(config, 'init')):
        usage()
        die()

    return config



# -----------------------------------------------------------------------------
def save_last_config(config):

    d = config.to_dict(False)

    work_dir = config.work_dir

    with open(os.path.join(work_dir, 'last_config.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(d, indent=4))










# -----------------------------------------------------------------------------
CHECKPOINT_VERSION = 1

def checkpoint_load(path_prefix, load_optimizer_state):
    """ """

    model_state_dict = torch.load(path_prefix + ".pt")
    if load_optimizer_state:
        optimizer_state_dict = torch.load(path_prefix + ".opti")
    else:
        optimizer_state_dict = None

    with open(path_prefix + '.json', 'r', encoding='utf-8') as f:
        js = f.read()
    j = json.loads(js)

    return (model_state_dict, optimizer_state_dict,
            j['train'],
            j['sample'],

            j['model'],
            j['dataset'],
            j['trainer'] )



def checkpoint_save(path_prefix, 
                    model, optimizer, 

                    train_config_dict,
                    sample_config_dict,

                    model_config_dict,
                    dataset_config_dict,
                    trainer_config_dict):

    # no CTRL+C interruptions while saving, please (malformed checkpoint files)
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


    torch.save(model.state_dict(), path_prefix + ".pt")
    torch.save(optimizer.state_dict(), path_prefix + ".opti")

    config_info = {'_version': CHECKPOINT_VERSION,
                   'train': train_config_dict,
                   'sample': sample_config_dict,

                   'model': model_config_dict,
                   'dataset': dataset_config_dict,
                   'trainer': trainer_config_dict
                   }

    json_str = json.dumps(config_info, indent=4)

    with open(path_prefix + '.json', 'w', encoding='utf-8') as f:
        f.write(json_str)


    # restore original handler
    signal.signal(signal.SIGINT, original_sigint)



def checkpoint_exists(path_prefix):

    return ( os.path.isfile(path_prefix + ".pt") and 
             os.path.isfile(path_prefix + ".opti") and 
             os.path.isfile(path_prefix + ".json") )

