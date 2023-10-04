"""

"""

import os, sys, copy, signal, json
from enum import IntFlag

import torch

from .tokendataset import GPT2TokensDataset
from .chardataset import CharDataset, CharLineDataset

from .conf import Conf



# -----------------------------------------------------------------------------
"""
How config works:

1) User code sets values on a config object got from empty_config().

2) Any command line args are merged into this config when calling config_run(), or by using merge_config_from_sysargv().

3) When initializing (init_new(), init_pretrained(), init_resume(), this config is passed and overrides defaults.

A call to init_new(), init_pretrained(), init_resume() sets the initial config. 

Later calls to sample() or train() can include parameters that will set config settings, but these are local to the function - the global config is the same when returning.

Config settings (read from command line or in python) can be set with these types of values:
    None
    int
    float
    str

No booleans: use 0 or 1 instead

"""


def empty_config():
    """ an empty config: options that are later set will override the full config """
    c = Conf()

    # sample
    c.sample = Conf()

    # train
    c.train = Conf()


    # dataset
    c.dataset = Conf()

    # model
    c.model = Conf()

    # trainer
    c.trainer = Conf()

    return c



def full_default_config():
    """ returns a full config with all possible values, type information and help"""

    from .sample import Sample
    from .train import Train
    from .model import GPT
    from .trainer import Trainer


    c = empty_config()

    c.seed = -1 # 0 means initial random seed, -1 means don't set seed


    # sample
    c.sample = Sample.get_default_config()

    # train
    c.train = Train.get_default_config()


    # dataset
    c.dataset = dataset_get_default_config()

    # model
    c.model = GPT.get_default_config()

    # trainer
    c.trainer = Trainer.get_default_config()

    return c



# -----------------------------------------------------------------------------
def dataset_get_default_config():

    c = Conf()

    c.setup('class_name', None, str, 'Dataset type id: ' + ','.join(DATASET_CLASS_MAP.keys()))
    
    c.setup('train_path', None, str, 'Train dataset path')

    c.setup('train_split', 0.9, float, 'Train dataset split ratio (0..1) for creating a validation dataset. Only used if val_path is unset')

    c.setup('val_path', None, str, 'Validation dataset path. If set, train_split is not used')

    c.setup('params', None, str, "String in the form 'name=vale,name=value,...' containing extra parameters for dataset creation")

    return c




DATASET_CLASS_MAP = {'gpt2': GPT2TokensDataset, 'char': CharDataset, 'charline': CharLineDataset}

def dataset_class_from_name(class_name):
    return DATASET_CLASS_MAP[class_name]






# -----------------------------------------------------------------------------
def merge_config_from_sysargv(sys_argv, base_config=None):

    argv = sys_argv[1:]

    if base_config is not None:
        config = base_config
    else:
        config = empty_config()

    config.update_from_args(argv, key_must_exist=False)

    return config







# -----------------------------------------------------------------------------
CHECKPOINT_VERSION = 1

def checkpoint_load(path_prefix, load_optimizer_state:bool):
    """ """

    model_state_dict = torch.load(path_prefix + "model.pt")
    if load_optimizer_state:
        optimizer_state_dict = torch.load(path_prefix + "optimizer.pt")
    else:
        optimizer_state_dict = None

    with open(path_prefix + 'state.json', 'r', encoding='utf-8') as f:
        js = f.read()
    j = json.loads(js)

    return (j['state'], j['config'],
            model_state_dict, optimizer_state_dict)



def checkpoint_save(path_prefix, 
                    state_dict, config_dict,
                    model_state_dict, 
                    optimizer_state_dict
                    ):

    # no CTRL+C interruptions while saving to avoid incomplete/corrupt checkpoint files
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    torch.save(model_state_dict, path_prefix + "model.pt")
    torch.save(optimizer_state_dict, path_prefix + "optimizer.pt")

    # config json
    d = {'state': state_dict,
         'config': config_dict,
         '_version': CHECKPOINT_VERSION}

    json_str = json.dumps(d, indent=2)

    with open(path_prefix + 'state.json', 'w', encoding='utf-8') as f:
        f.write(json_str)


    # restore original handler
    signal.signal(signal.SIGINT, original_sigint)




def checkpoint_exists(path_prefix):
    return ( os.path.isfile(path_prefix + "model.pt") and 
             os.path.isfile(path_prefix + "optimizer.pt") and 
             os.path.isfile(path_prefix + "state.json") )





LOSS_LOG_FILENAME = 'loss.csv'

def loss_load(path_prefix):
    with open(path_prefix + LOSS_LOG_FILENAME, 'r') as f:
        lines = [line[:-1] for line in f]

    out = []

    for line in lines:
        t=line.split(',')
        out.append(t)

    return out


def loss_append(path_prefix, loss_list):

    out = []
    for t in loss_list:
        if len(t)==3 and t[2] == float('inf'): # (iter,train,inf) -> (iter,train)
            t = t[:-1]
        out.append(','.join(map(str,t)))
    loss_text = '\n'.join(out)

    with open(path_prefix + LOSS_LOG_FILENAME, 'a') as f:
        f.write(loss_text + '\n')



def loss_trim(path_prefix, last_iter_num):

    path = path_prefix + LOSS_LOG_FILENAME

    if last_iter_num is None: # delete all
        with open(path, 'w') as f:
            f.write('')
        return

    try:
        l = loss_load(path)
    except OSError:
        return

    loss_list = []

    for i in range(len(l)-1, -1, -1):
        t = int(l[i][0])

        if t <= last_iter_num: # found first <= iter_num: keep up till this one
            loss_list = l[:i+1]
            break

    out = []
    for t in loss_list:
        out.append(','.join(map(str,t)))
    loss_text = '\n'.join(out)

    with open(path, 'w') as f:
        f.write(loss_text)
        if len(loss_text):
            f.write('\n')






# -----------------------------------------------------------------------------
class LogFlag(IntFlag):
    NONE = 0

    INIT = 1 # init messages in init_* and in sample, train, etc.

    SAMPLE = 2

    TRAIN_ITER = 4
    TRAIN_EVAL = 8
    TRAIN_DOT = 16
    TRAIN = TRAIN_DOT | TRAIN_ITER | TRAIN_EVAL

    CUDA_MEMORY = 256

    ALL = INIT | SAMPLE | TRAIN | CUDA_MEMORY



