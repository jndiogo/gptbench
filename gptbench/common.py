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


# -----------------------------------------------------------------------------
"""
Config overriding order:

1) Initial config = common.default_config (full config with all values set)

2) User script sets values into initial config

3) If resume or init from gpt-2: config is overriden by saved / pretrained values

4) Config is then overriden from post_config which is usually populated from sys.argv

5) Config is then ready to be used to load datasets, create model, trainer, etc

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
CHECKPOINT_VERSION = 1

def checkpoint_load(path_prefix):
    """ """

    model_state_dict = torch.load(path_prefix + ".pt")
    optimizer_state_dict = torch.load(path_prefix + ".opti")

    with open(path_prefix + '.json', 'r', encoding='utf-8') as f:
        js = f.read()
    j = json.loads(js)

    return (model_state_dict, optimizer_state_dict, 
            j['model'], 
            j['trainer'],
            j['dataset'],
            j['eval'], j['eval_iters'], j['_iter_num'], j['_loss'] )



def checkpoint_save(path_prefix, 
                    model, optimizer, 
                    model_config_dict,
                    trainer_config_dict,
                    dataset_config_dict,
                    eval, eval_iters, _iter_num, _loss):

    torch.save(model.state_dict(), path_prefix + ".pt")
    torch.save(optimizer.state_dict(), path_prefix + ".opti")

    config_info = {'version': CHECKPOINT_VERSION,
                   'model': model_config_dict,
                   'trainer': trainer_config_dict,
                   'dataset': dataset_config_dict,
                   'eval': eval, 'eval_iters': eval_iters,
                   '_iter_num': _iter_num, '_loss': _loss                   
                   }

    json_str = json.dumps(config_info, indent=4)

    with open(path_prefix + '.json', 'w', encoding='utf-8') as f:
        f.write(json_str)








# -----------------------------------------------------------------------------
def train(config, trainer, start_loss=float('inf')):
    """config is global config """

    assert (config.eval & 3) != 0, "config.eval must be set to 1, 2 or 1|2"

    model = trainer.model
    train_dataset = trainer.train_dataset

    last_saved_loss = start_loss

    model.train()


    # iteration callback
    def batch_end_callback(trainer):
        nonlocal last_saved_loss

        iter_num = trainer.iter_num

        # report, save model?
        if iter_num > trainer.get_start_iter_num():

            model_evaluated = False

            if iter_num % config.eval_period == 0: # evaluate loss 

                train_loss, val_loss = estimate_loss(
                    train_dataset,
                    config.dataset.val,
                    model,
                    trainer.config.batch_size,
                    config.eval_iters)

                if config.eval & 3 == 3:
                    loss = (train_loss + val_loss) / 2.
                else:
                    loss = val_loss if (config.eval & 2) and val_loss else train_loss

                val_loss = val_loss if val_loss is not None else float('inf')

                print(f"iter {iter_num} ({trainer.epoch_from_iter_num():.3f} epoch) | loss {loss:.4f} ({train_loss:.4f},{val_loss:.4f}) | iter_dt {trainer.iter_dt * 1000:.2f}ms")

                model_evaluated = True


                if loss < last_saved_loss: # save a checkpoint

                    print(f"==> Saving model at loss={loss:.4f} iter={iter_num}")

                    checkpoint_save(config._model_path_prefix, 
                                    model, trainer.optimizer,
                                    
                                    config.model.to_dict(False, GPT.checkpoint_config_keys()), 
                                    config.trainer.to_dict(False, Trainer.checkpoint_config_keys()),
                                    config.dataset.to_dict(False, DatasetBase.checkpoint_config_keys()),
                                    
                                    config.eval, config.eval_iters, 
                                    iter_num, loss)

                    last_saved_loss = loss


            if iter_num % config.eval_sample_period == 0:
                # evaluate both the train and test score
                sample(config.sampler, model, train_dataset)

                model_evaluated = True

            
            if model_evaluated:
                model.train() # revert model to training mode


        print('.', end='', flush=True)

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()




@torch.no_grad()
def estimate_loss(train_dataset, val_dataset, model, batch_size, iters):
    """ train_dataset or val_dataset can be None to skip its eval returns train_loss,val_loss any of which can be None"""

    model.eval()

    out = []

    for split in ['train', 'val']:
        dataset=train_dataset if split == 'train' else val_dataset

        if dataset is None:
            out.append(None)
            continue

        losses = torch.zeros(iters)

        for k in range(iters):

            ix = torch.randint(len(dataset), (batch_size,))

            batches = [dataset[i] for i in ix] # [(x,y),(x,y),...]

            x = torch.stack([x for x,_ in batches])
            y = torch.stack([y for _,y in batches])

            x, y = x.to(model.device), y.to(model.device)

            _, loss = model(x,y)

            losses[k] = loss.item()

        out.append(losses.mean().item())

    return out




# -----------------------------------------------------------------------------
@torch.no_grad()
def sample(sampler_config, model, train_dataset, stop_asap=None):
    """ stop_asap=[False] - when set to True, must break and return """

    count=sampler_config.count
    length=sampler_config.len
    context=sampler_config.start
    top=sampler_config.top
    temp=sampler_config.temp
    token_emit=sampler_config.pertoken
    eotstop=sampler_config.eotstop
    eot_token = train_dataset.get_eot_token()

    model.eval()

    ix = train_dataset.encode(context)    
    x = torch.tensor(ix, dtype=torch.long).to(model.device)

    if token_emit:

        def emit(idx):

            idx=idx[0].tolist()

            is_eot = idx[0] == eot_token

            if is_eot and eotstop==-1:
                return -1

            chars = train_dataset.bufd_decode(idx)
            print(chars, sep='', end='', flush=True)

            if is_eot and eotstop==1:
                return -1

            return 0


        x = x.repeat([1, 1])

        for _ in range(count):
            print_sepline()
            print(context, sep='', end='')

            model.generate(x, length, temperature=temp, do_sample=True, top=top, 
                           token_callback = emit if token_emit else None,
                           stop_asap=stop_asap)

            if stop_asap is not None and stop_asap[0]:
                return

            # flush any buffered utf-8 characters
            chars = train_dataset.bufd_flush()
            print(chars, sep='', end='', flush=True)

            print()


    else:
        x = x.repeat([count, 1])
        y = model.generate(x, length, temperature=temp, do_sample=True, top=top, 
                           token_callback = emit if token_emit else None,
                           stop_asap=stop_asap)

        if stop_asap is not None and stop_asap[0]:
            return

        for ir in range(y.size(0)):
            row = y[ir,:].tolist()

            if eotstop:
              index = row.index(eot_token)
              if index >= 0:
                row = row[:index if eotstop==-1 else index+1]

            completion = train_dataset.decode(row)

            print_sepline()
            print(completion)





# -----------------------------------------------------------------------------
@torch.no_grad()
def prompt(config, model, train_dataset):
    """ """

    allowed_cmds = [
    'seed',
    'help',
    'quit',
    'config',

    'start',
    'count',
    'len',
    #'start',
    'pertoken',
    'eotstop',
    'top',
    'temp',
    'multiline',
    ]


    sampler_config = config.sampler

    sampler_config = copy.copy(sampler_config)
    sampler_config.token_emit = 1


    stop_asap = [False]

    def signal_handler(signal, frame):
        nonlocal stop_asap
        print('\n<stopping>')
        stop_asap[0] = True

    original_sigint = signal.getsignal(signal.SIGINT)


    def print_help():
        print("Enter sampling start text or a command in the form -cmd or -cmd=val. Possible commands: ", ['-' + c for c in allowed_cmds], "\n Press Ctrl+C once to stop generation.")

    while True:
        p = ''
        if sampler_config.multiline:
            prompt='V\n'
        else:
            prompt='> '

        while True:
            try:
                p += input(prompt)
            except EOFError:
                break

            if not sampler_config.multiline:
                break
            else:
                p += '\n'
                prompt=''

        if not len(p):
            continue


        if p.startswith('-'): # a command
            p = p.strip()
            cmds = p.split(' ')

            quit = False
            for c in cmds:
                if c.startswith('-'):
                    c = c[1:]
                    if c[:1] == '-': c = c[1:] # strip eventual second -

                if '=' in c:
                    kv = list(c.split('='))
                else: # -cmd -> -cmd=1
                    kv = [c,'1']

                k,v=kv

                if not k in allowed_cmds:
                    print_help()
                    continue

                if k == 'help':
                    print_help()
                    break
                elif k == 'quit':
                    print("Quitting")
                    quit = True
                    break
                elif k == 'seed':
                    set_seed(int(v))
                elif k == 'config':
                    print("Config:")
                    print(config)
                else:
                    cmd_list = [ '-' + k + '=' + v ]
                    sampler_config.merge_from_args(cmd_list, key_must_exist=True)

            if quit:
                break
        else:
            p = p.replace("\\n", "\n")
            sampler_config.start=p

            stop_asap = [False]
            signal.signal(signal.SIGINT, signal_handler)

            sample(sampler_config, model, train_dataset, stop_asap=stop_asap)

            signal.signal(signal.SIGINT, original_sigint)

            print_sepline()



def load_datasets(dataset_config, block_size, to_train):

    if not to_train:
        dataset_config.trainsplit = 1. # when sampling force train split to dummy 1, to avoid errors creating small val 

    return DatasetBase.create_train_val_datasets(block_size,
                                                 dataset_config.cls,
                                                 dataset_config.trainsplit, 
                                                 data_path=dataset_config.path)





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
def config_from_sysargv(sys_argv):

    argv = sys_argv[1:]

    if len(argv) < 2: # at least mode, init
        usage_exit()

    c = empty_config()

    c.merge_from_args(argv, key_must_exist=False)

    if not hasattr(c, 'mode') or not hasattr(c, 'init'):
        usage_exit()

    return c





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

    #default
    if not hasattr(config, 'name'):
        config.name = 'model'

    assert hasattr(config, 'mode') and hasattr(config, 'init'), "mode and init are mandatory config values"


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

        print(f"Batches per epoch: {int(trainer.batch_count_for_epoch())}")

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

