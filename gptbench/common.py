"""

"""

import os, sys, copy, signal, json

from gptbench.model import GPT
from gptbench.trainer import Trainer
from gptbench.dataset import DatasetBase
from gptbench.train import train, checkpoint_load
from gptbench.sample import sample, prompt
from gptbench.utils import CfgNode, set_seed, last_config_save, die, print_sepline, cuda_max_memory_init, cuda_max_memory_print



# -----------------------------------------------------------------------------
"""
How config works:

1) User code sets values on a config object got from empty_config()

2) Any command line args are merged into that config with merge_config_from_sysargv()

3) This partial config is then passed into run(). It's merged as needed into a default full config and any resumed checkpoint information, to get the final fully resolved config


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


def default_full_config():
    """ returns a full config with all possible values """
    c = empty_config()

    c.work_dir = './out'

    c.seed = 0 # 0 means random seed

    c.eval = 2 # how to estimate loss -> 1: on test data, 2: on val data (or test if no val dataset), 1|2=3: mean(test,val)
    c.eval_iters = 100

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

def merge_config_from_sysargv(sys_argv, base_config = None):

    argv = sys_argv[1:]

    if len(argv) < 2: # at least mode, init
        usage_exit()

    if base_config is not None:
        config = base_config
    else:
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
        dataset_config.train_split = 1. # when sampling force train split to dummy 1, to avoid errors creating small val 

    return DatasetBase.create_train_val_datasets(block_size,
                                                 dataset_config.cls,
                                                 dataset_config.train_split, 
                                                 data_path=dataset_config.path)




# -----------------------------------------------------------------------------
def run(part_config):

    """ part_config is a partial config containing only settings that will override defaults or resumed values """

    # mandatory mode, init [name] params
    assert hasattr(part_config, 'mode') and hasattr(part_config, 'init'), "mode and init are mandatory config values"

    # defaults
    if not hasattr(part_config, 'name'):
        part_config.name = 'model'

    if not hasattr(part_config, 'work_dir'):
        part_config.work_dir = './out'


    # setup_work_dir
    work_dir = part_config.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # out/name .pt, .json, etc
    part_config._model_path_prefix = os.path.join(work_dir, part_config.name)



    if part_config.mode == 'train':
        do_train = True

    elif part_config.mode == 'sample' or part_config.mode == 'prompt':
        do_train = False

        # force 0 dropout when sampling
        part_config.model.dropout=0.

        # force per-token emission if prompt mode
        if part_config.mode == 'prompt':
            part_config.sampler.token_emit = True

    else:
        die('Config -mode= must be one of: train, sample, prompt')


    config = default_full_config()

    # resolve seed beforehand
    seed = part_config.seed if hasattr(part_config, 'seed') else config.seed
    set_seed(config.seed)

    start_iter_num = 0
    start_loss = float('inf') 
    optimizer_state_dict = None

    # model init
    if part_config.init == 'new' or part_config.init == 'resume':

        if part_config.init == 'new':
            print(f"Initializing new model {part_config._model_path_prefix}")

            model_state_dict = None

        else: # load checkpoint
            print(f"Loading checkpoint from {part_config._model_path_prefix}")

            (model_state_dict, optimizer_state_dict, 
             model_config_dict,
             trainer_config_dict,
             dataset_config_dict,
             config.eval, config.eval_iters, config.eval_period, config.eval_sample_period,
             start_iter_num, start_loss) = checkpoint_load(part_config._model_path_prefix, do_train)
            # only load optimizer state if do_train

            # merge resumed configs into config
            config.model.merge_from_dict(model_config_dict, GPT.checkpoint_config_keys())

            config.trainer.merge_from_dict(trainer_config_dict, Trainer.checkpoint_config_keys())

            config.dataset.merge_from_dict(dataset_config_dict, DatasetBase.checkpoint_config_keys())

            # if resumed dataset file is no longer available: erase it - either part_config's or an empty dummy will be used
            if not os.path.isfile(config.dataset.path):
                config.dataset.path = None


            print(f"Checkpoint: iter={start_iter_num}, loss={start_loss}")


        # merge part_config into config for the final resolved config
        config.merge_from_config(part_config)

        # load datasets
        (config.dataset.train, config.dataset.val) = load_datasets(config.dataset, config.model.block_size, do_train)

        # ensure right vocab_size
        config.model.vocab_size = config.dataset.train.get_vocab_size()

        model = GPT(config.model)

        if model_state_dict is not None:
            model.load_state_dict( model_state_dict )


    elif part_config.init.startswith('gpt'):
        if part_config.init not in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            die('Config -init= must be one of: new, resume, gpt2, gpt2-medium, gpt2-large, gpt2-xl')

        print(f"Initializing model from {part_config.init}")

        # merge part_config into config for the final resolved config
        config.merge_from_config(part_config)

        # will set config.model.* parameters as needed
        model, config.model = GPT.from_pretrained(config.init, config.model)

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



    # config is now resolved: save a copy
    last_config_save(config)


    # log config which is now fully resolved
    print("Model params: %.2fM" % (model.get_num_params()/1e6,))
    print(f"Running on device: {model.device} dtype: {model.dtype}")

    print_sepline()
    print('Resolved config:')
    print(config)
    print_sepline()


    if do_train: # training
        print("Train mode")
        train(config, trainer, start_loss)

    elif config.mode == 'prompt':
        print("Prompt mode: press Enter (single line mode), or Ctrl+D / Ctrl+Z (multiline mode) to submit starting text. -help for available -commands:")
        prompt(config, model, config.dataset.train)

    else: # sampling
        print("Sampling mode")
        sample(config.sampler, model, config.dataset.train)

