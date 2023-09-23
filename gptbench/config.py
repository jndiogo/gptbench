"""

"""

import os, sys, copy, signal, json
from enum import IntFlag

import torch


from .utils import CfgNode



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
    c = CfgNode()

    # sample
    c.sample = CfgNode()

    # train
    c.train = CfgNode()


    # dataset
    c.dataset = CfgNode()

    # model
    c.model = CfgNode()

    # trainer
    c.trainer = CfgNode()

    return c



def default_full_config():
    """ returns a full config with all possible values """

    from .sample import Sample
    from .train import Train
    from .model import GPT
    from .trainer import Trainer


    c = empty_config()

    c.seed = 0 # 0 means random seed


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

    if base_config is not None:
        config = base_config
    else:
        config = empty_config()

    config.merge_from_args(argv, key_must_exist=False)

    return config







# -----------------------------------------------------------------------------
CHECKPOINT_VERSION = 1

def checkpoint_load(path_prefix, load_optimizer_state):
    """ """

    model_state_dict = torch.load(path_prefix + "model.pt")
    if load_optimizer_state:
        optimizer_state_dict = torch.load(path_prefix + "optimizer.pt")
    else:
        optimizer_state_dict = None

    with open(path_prefix + 'config.json', 'r', encoding='utf-8') as f:
        js = f.read()
    j = json.loads(js)

    return (model_state_dict, optimizer_state_dict,
            j['sample'],
            j['train'],

            j['model'],
            j['dataset'],
            j['trainer'] )



def checkpoint_save(path_prefix, 
                    model, optimizer,

                    sample_config_dict,
                    train_config_dict,

                    model_config_dict,
                    dataset_config_dict,
                    trainer_config_dict):
    """
    loss is in the form [(iter_num,train_loss[,val_loss]),...]
    """


    # no CTRL+C interruptions while saving to avoid incomplete/corrupt checkpoint files
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


    torch.save(model.state_dict(), path_prefix + "model.pt")
    torch.save(optimizer.state_dict(), path_prefix + "optimizer.pt")

    # config json
    config_info = {'_version': CHECKPOINT_VERSION,
                   'train': train_config_dict,
                   'sample': sample_config_dict,

                   'model': model_config_dict,
                   'dataset': dataset_config_dict,
                   'trainer': trainer_config_dict
                   }

    json_str = json.dumps(config_info, indent=4)

    with open(path_prefix + 'config.json', 'w', encoding='utf-8') as f:
        f.write(json_str)


    # restore original handler
    signal.signal(signal.SIGINT, original_sigint)



def checkpoint_exists(path_prefix):

    return ( os.path.isfile(path_prefix + "model.pt") and 
             os.path.isfile(path_prefix + "optimizer.pt") and 
             os.path.isfile(path_prefix + "config.json") )





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



