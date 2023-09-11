"""

"""

import os, sys, copy, signal, json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from gptbench.model import GPT
from gptbench.trainer import Trainer
from gptbench.dataset import GPT2TokensDataset, DatasetBase
from gptbench.utils import CfgNode, set_seed, last_config_save, die, print_sepline

from gptbench.train import train, checkpoint_load
from gptbench.sample import sample, prompt




# -----------------------------------------------------------------------------
"""
How config works:

1) Start with an initial config = common.default_config() - creates a full config with all values set

2) Script sets any base values into the initial config

3) If resuming from a checkpoint or initializing from pretrained gpt-2: some config values are overriden by information in the saved checkpoint

4) Config is then overriden by any values in post_config: tipically from command line args

"""


def empty_config():
    c = CfgNode()

    # model
    c.model = CfgNode()

    # trainer
    c.trainer = CfgNode()

    # dataset
    c.dataset = CfgNode()

    # sampler
    c.sampler = CfgNode()

    return c


def default_config():
    """ returns a full config with all possible values """
    c = empty_config()

    c.work_dir = './out'

    c.seed = 0 # 0 means random seed

    c.eval = 2 # how to estimate loss -> 1: on test data, 2: on val data (or test if no val dataset), 1|2=3: mean(test,val)
    c.eval_iters = 10

    c.eval_period = 100 # each n batches we eval and check if saving model
    c.eval_sample_period = 1000


    # model
    c.model = GPT.get_default_config()

    # trainer
    c.trainer = Trainer.get_default_config()

    # dataset
    c.dataset = DatasetBase.get_default_config()

    # sampler
    c.sampler = CfgNode()
    c.sampler.count = 1
    c.sampler.len = 400
    c.sampler.start = ' '
    c.sampler.pertoken = 1 # display each token immediately
    c.sampler.eotstop = 0 # 0 don't stop, -1 stop before, 1 stop after (and display it)
    c.sampler.top = 40 # 0..1: top_p, > 1: top_k
    c.sampler.temp = 1. # temperature
    c.sampler.multiline = 0 # input multiple lines until a Ctrl+D or Ctrl+Z (in Windows)

    return c





# -----------------------------------------------------------------------------

def config_from_sysargv(sys_argv):

    argv = sys_argv[1:]

    if len(argv) < 2: # at least mode, init
        usage_exit()

    config = empty_config()
    config.merge_from_args(argv, key_must_exist=False)

    if not (hasattr(config, 'mode') and hasattr(config, 'init')):
        usage_exit()

    return config



# -----------------------------------------------------------------------------
def usage_exit():
    die('''
Usage: run.py mode=train init=resume [name=model21] [config] ...

   mode=train|sample|prompt
   init=new|resume|gpt2*, gpt2* is one of gpt2, gpt2-medium, gpt2-large, gpt2-xl
   name=project name to save and resume chepckpoints

[config] options are in the form -name, -name=value or -area.name=value and override any entries of the same name in config.

    -model.*=model config options
    -trainer.*=trainer config options
    -sampler=options associated with model sampling/prompting
    -dataset.path=path to training dataset, default is "" for dummy dataset
    ''')



# -----------------------------------------------------------------------------
def load_datasets(dataset_config, block_size, to_train):

    if not to_train:
        dataset_config.trainsplit = 1. # when sampling force train split to dummy 1, to avoid errors creating small val 

    return DatasetBase.create_train_val_datasets(block_size,
                                                 dataset_config.cls,
                                                 dataset_config.trainsplit, 
                                                 data_path=dataset_config.path)




# -----------------------------------------------------------------------------
def run(config, post_config = None):

    """ post_config is used after a resume or pretrained initialization to override such params (or any other). usually post_config comes from sys.argv """


    # get mandatory mode, init [name] params
    if post_config is not None:
        if hasattr(post_config, 'mode'):
            config.mode = post_config.mode
        if hasattr(post_config, 'init'):
            config.init = post_config.init
        if hasattr(post_config, 'name'):
            config.name = post_config.name

    assert hasattr(config, 'mode') and hasattr(config, 'init'), "mode and init are mandatory config values"

    # default name is 'model'
    if not hasattr(config, 'name'):
        config.name = 'model'



    # setup_work_dir
    work_dir = config.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # out/name .pt, .json, etc
    config._model_path_prefix = os.path.join(work_dir, config.name)


    set_seed(config.seed)


    if config.mode == 'train':
        do_train = True

    elif config.mode == 'sample' or config.mode == 'prompt':
        do_train = False

        config.model.dropout=0.

        if config.mode == 'prompt':
            config.sampler.token_emit = True

    else:
        die('Config -mode= must be one of: train, sample, prompt')


    start_iter_num = 0
    start_loss = float('inf') 
    optimizer_state_dict = None

    # model init
    if config.init == 'new' or config.init == 'resume':

        if config.init == 'new':
            print(f"Initializing new model {config._model_path_prefix}")

            model_state_dict = None

        else: # load checkpoint
            print(f"Loading checkpoint from {config._model_path_prefix}")

            (model_state_dict, optimizer_state_dict, 
             model_config_dict,
             trainer_config_dict,
             dataset_config_dict,
             config.eval, config.eval_iters,
             start_iter_num, start_loss) = checkpoint_load(config._model_path_prefix)

            # merge resumed configs
            config.model.merge_from_dict(model_config_dict, GPT.checkpoint_config_keys())

            config.trainer.merge_from_dict(trainer_config_dict, Trainer.checkpoint_config_keys())

            config.dataset.merge_from_dict(dataset_config_dict, DatasetBase.checkpoint_config_keys())

            # if resumed dataset file is no longer available: erase it - an empty dummy will be used
            if not os.path.isfile(config.dataset.path):
                config.dataset.path = None


            print(f"Checkpoint: iter={start_iter_num}, loss={start_loss}")

        if post_config is not None:
            config.merge_from_config(post_config)

        # load datasets
        (config.dataset.train, config.dataset.val) = load_datasets(config.dataset, config.model.block_size, do_train)

        # ensure right vocab_size
        config.model.vocab_size = config.dataset.train.get_vocab_size()


        model = GPT(config.model)

        if model_state_dict is not None:
            model.load_state_dict( model_state_dict )


    elif config.init.startswith('gpt'):
        if config.init not in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            die('Config -init= must be one of: new, resume, gpt2, gpt2-medium, gpt2-large, gpt2-xl')

        print(f"Initializing model from {config.init}")

        model, config.model = GPT.from_pretrained(config.init, config.model)

        if post_config:
            config.merge_from_config(post_config)

        # create a training dataset: possibly dummy with one sample
        (config.dataset.train, config.dataset.val) = load_datasets(config.dataset, config.model.block_size, do_train)

        # ensure right vocab_size
        config.model.vocab_size = config.dataset.train.get_vocab_size()

    else:
        die('Config -init= must be one of: new, resume, gpt2, gpt2-medium, gpt2-large, gpt2-xl')



    print(f"Dataset: {config.dataset.path if config.dataset.path else 'dummy empty dataset'}")
   

    if do_train: ## training, not sampling
        # construct the trainer object
        trainer = Trainer(config.trainer, model, config.dataset, start_iter_num=start_iter_num)

        if optimizer_state_dict is not None:
            print("Resuming optimizer state")
            trainer.set_optimizer_state_dict(optimizer_state_dict)

        print(f"Batches per epoch: {int(trainer.batches_for_epoch())}")

        # only worth checking for training
        if config.dataset.val is None:
            config.eval &= 1 # clear val bit
        if config.eval & 3 == 0: # force at least train
            config.eval = 1



    last_config_save(config)

    # log config which is now fully resolved
    print("Running on device", model.device)
    print("Model params: %.2fM" % (model.get_num_params()/1e6,))

    print_sepline()
    print('Config:')
    print(config)
    print_sepline()


    if do_train: # training
        print("Train mode")
        train(config, trainer, start_loss)

    elif config.mode == 'prompt':
        print("Prompt mode: to submit press Enter (single line mode), or Ctrl+D / Ctrl+Z (multiline mode):")
        prompt(config, model, config.dataset.train)

    else: # sampling
        print("Sampling mode")
        sample(config.sampler, model, config.dataset.train)

