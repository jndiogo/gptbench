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

Later calls to sample() or train() can include parameters that will set config options, but these are local to the function - the global config is the same when returning.

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

    model_state_dict = torch.load(path_prefix + ".pt")
    if load_optimizer_state:
        optimizer_state_dict = torch.load(path_prefix + ".opti")
    else:
        optimizer_state_dict = None

    with open(path_prefix + '.json', 'r', encoding='utf-8') as f:
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








# -----------------------------------------------------------------------------
class LogFlag(IntFlag):
    NONE = 0

    INIT = 1

    SAMPLE = 2

    BATCH_DOT = 4
    BATCH_LOSS = 8
    EVAL_LOG = 16
    TRAIN = BATCH_DOT | BATCH_LOSS | EVAL_LOG

    CUDA_MEMORY = 256

    ALL = INIT | SAMPLE | BATCH_DOT | BATCH_LOSS | EVAL_LOG | CUDA_MEMORY



