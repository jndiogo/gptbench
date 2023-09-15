"""

"""

import os, sys, copy, signal, json

from gptbench.train import train as _train, checkpoint_load, train_get_default_config, train_checkpoint_config_keys, train_config_resolve
from gptbench.sample import sample as _sample, prompt as _prompt, sample_get_default_config, sample_config_resolve, sample_checkpoint_config_keys

from gptbench.model import GPT
from gptbench.trainer import Trainer
from gptbench.dataset import dataset_get_default_config, dataset_checkpoint_config_keys, dataset_class_from_name

from gptbench.utils import CfgNode, set_seed, save_last_config, die, print_sepline, cuda_max_memory_init, cuda_max_memory_print



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


    # model
    c.model = CfgNode()

    # trainer
    c.trainer = CfgNode()

    # dataset
    c.dataset = CfgNode()


    return c


def default_full_config():
    """ returns a full config with all possible values """

    c = empty_config()

    c.seed = 0 # 0 means random seed

    c.work_dir = './out'

    c.verbose = 2 # 2: display all initial info, 1: just display resolved config, 0: no intial info

    # train
    c.train = train_get_default_config()

    # sample
    c.sample = sample_get_default_config()


    # model
    c.model = GPT.get_default_config()

    # trainer
    c.trainer = Trainer.get_default_config()

    # dataset
    c.dataset = dataset_get_default_config()


    return c





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
def usage():
    print('''Usage: run.py mode=train init=resume [name=model21] [config] ...

   mode=train|sample|prompt
   init=new|resume|gpt2*, gpt2* is one of gpt2, gpt2-medium, gpt2-large, gpt2-xl
   name=project name to save and resume chepckpoints

[config] options are in the form -name, -name=value or -area.name=value and override any entries of the same name in config.

    -train=options associated with training
    -sample=options associated with sampling/prompting

    -model.*=model config options
    -trainer.*=trainer config options
    -dataset.train_path=path to training dataset, None for dummy dataset

Default config:
''' + str(default_full_config()) )


def printv(str, min_verbose, verbose):
    if verbose >= min_verbose:
        print(str)



# -----------------------------------------------------------------------------
def load_datasets(dataset_config, block_size, to_train):

    if not to_train:
        dataset_config.val_path_or_train_split = 1. # when sampling force train split to dummy 1, to avoid errors creating small val 
    try:
        cls = dataset_class_from_name(dataset_config.class_name)
    except KeyError:
        die("Unknown config value dataset.class_name")

    return cls.load_train_val_datasets(dataset_config.train_path,
                                       dataset_config.val_path_or_train_split,
                                       block_size,
                                       repeat_if_needed=True)





# -----------------------------------------------------------------------------
def train(part_config, name=None, init=None, verbose=None):

    part_config = copy.copy(part_config)

    part_config.mode = 'train'

    if name is not None:
        part_config.name = name

    if init is not None:
        part_config.init = init
    
    if verbose is not None:
        part_config.verbose = verbose

    return run(part_config)


def sample(part_config, name=None, init=None, verbose=None):

    part_config = copy.copy(part_config)

    part_config.mode = 'sample'    

    if name is not None:
        part_config.name = name

    if init is not None:
        part_config.init = init

    if not hasattr(part_config, 'init'):
        part_config.init = 'resume'

    assert part_config.init != 'new', "To sample, config.init must be set to 'resume' or to any of the gpt2 models"

    if not hasattr(part_config, 'verbose'):
        part_config.verbose = verbose

    if verbose is not None:
        part_config.verbose = verbose

    return run(part_config)


def prompt(part_config, name=None, init=None, verbose=None):

    part_config = copy.copy(part_config)

    part_config.mode = 'prompt'

    if name is not None:
        part_config.name = name

    if init is not None:
        part_config.init = init

    if not hasattr(part_config, 'init'):
        part_config.init = 'resume'

    assert part_config.init != 'new', "To prompt, config.init must be set to 'resume' or to any of the gpt2 models"

    if verbose is not None:
        part_config.verbose = verbose

    return run(part_config)




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

    else:
        die('Config -mode= must be one of: train, sample, prompt')


    config = default_full_config()

    # resolve beforehand
    verbose = part_config.verbose if hasattr(part_config, 'verbose') else config.verbose
    seed = part_config.seed if hasattr(part_config, 'seed') else config.seed

    set_seed(config.seed, verbose >= 2)


    optimizer_state_dict = None

    # model init
    if part_config.init == 'new' or part_config.init == 'resume':

        if part_config.init == 'new':
            printv(f"Initializing new model {part_config._model_path_prefix}", 2, verbose)

            model_state_dict = None


        else: # load checkpoint
            printv(f"Loading checkpoint from {part_config._model_path_prefix}", 2, verbose)

            (model_state_dict, optimizer_state_dict, 

             train_config_dict,
             sample_config_dict,

             model_config_dict,             
             dataset_config_dict,
             trainer_config_dict) = checkpoint_load(part_config._model_path_prefix, do_train)
            # only load optimizer state if do_train

            # merge resumed configs into config            
            config.train.merge_from_dict(train_config_dict, train_checkpoint_config_keys())
            config.sample.merge_from_dict(sample_config_dict, sample_checkpoint_config_keys())

            config.model.merge_from_dict(model_config_dict, GPT.checkpoint_config_keys())
            config.trainer.merge_from_dict(trainer_config_dict, Trainer.checkpoint_config_keys())
            config.dataset.merge_from_dict(dataset_config_dict, dataset_checkpoint_config_keys())

            # if resumed dataset file is no longer available: erase it - either part_config's or an empty dummy will be used
            #if not os.path.isfile(config.dataset.train_path):
            #    config.dataset.train_path = None

            printv(f"Checkpoint: iter_num={config.train.start_iter_num}, eval loss={config.train.start_eval_loss}", 2, verbose)


        # merge part_config into config for the final resolved config
        config.merge_from_config(part_config)

        # load datasets
        (train_dataset, val_dataset) = load_datasets(config.dataset, config.model.block_size, do_train)

        # ensure right vocab_size
        if part_config.init == 'resume':
            assert config.model.vocab_size == train_dataset.get_vocab_size(), f"Model vocab_size ({config.model.vocab_size} != Dataset vocab_size ({train_dataset.get_vocab_size()})"
        else:
            config.model.vocab_size = train_dataset.get_vocab_size()

        model = GPT(config.model)

        if model_state_dict is not None:
            model.load_state_dict( model_state_dict )


    elif part_config.init.startswith('gpt'):
        if part_config.init not in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            die('Config -init= must be one of: new, resume, gpt2, gpt2-medium, gpt2-large, gpt2-xl')

        printv(f"Initializing model from {part_config.init}", 2, verbose)

        # merge part_config into config for the final resolved config
        config.merge_from_config(part_config)

        # will set config.model.* parameters as needed
        model, config.model = GPT.from_pretrained(config.init, config.model)

        # create a training dataset: possibly dummy with one sample
        (train_dataset, val_dataset) = load_datasets(config.dataset, config.model.block_size, do_train)

        # ensure right vocab_size
        config.model.vocab_size = train_dataset.get_vocab_size()

    else:
        die('Config -init= must be one of: new, resume, gpt2, gpt2-medium, gpt2-large, gpt2-xl')


    printv(f"Dataset: {config.dataset.train_path if config.dataset.train_path else 'dummy empty dataset'} vocab_size: {train_dataset.get_vocab_size()}", 2, verbose)

   
    train_config_resolve(config, val_dataset)
    sample_config_resolve(config, train_dataset)


    # config is now resolved and definitive: save a copy
    save_last_config(config)


    # log config which is now fully resolved
    printv("Model params: %.2fM" % (model.get_num_params()/1e6,), 2, verbose)
    printv(f"Running on device: {model.device}, dtype: {model.dtype}", 2, verbose)

    if verbose >= 1:
        print_sepline()
        print('Resolved config:')
        print(config)
        print_sepline()


    if do_train: # training
        printv("Train mode:", 2, verbose)
        _train(config, model, 
              train_dataset, val_dataset,
              optimizer_state_dict=optimizer_state_dict)

    elif config.mode == 'prompt':
        print("Prompt mode: press Enter (single line mode), or Ctrl+D / Ctrl+Z (multiline mode) to submit starting text. Enter -help for available commands.")
        _prompt(config, model, train_dataset)

    else: # sampling
        printv("Sampling mode:", 2, verbose)
        _sample(config.sample, model, train_dataset)

