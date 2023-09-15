"""

"""

import os, sys, copy, signal, json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from gptbench.model import GPT
from gptbench.trainer import Trainer
from gptbench.utils import CfgNode, print_sepline, set_seed





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


def sample_config_resolve(config, train_dataset):
    """ config is global_config """

    # check sample.start
    if config.sample.start is not None:
        if not train_dataset.is_text_valid(config.sample.start):
            print(f"Config sample.start is not valid for dataset's vocabulary. Set to None (random)")
            config.sample.start = None # random vocab item on each sampling

    if config.sample.top > config.model.vocab_size:
        print(f'Config sample.top only up to vocab_size: {config.model.vocab_size}')
        config.sample.top = config.model.vocab_size

    # force per-token emission if prompt mode
    if config.mode == 'prompt':
        config.sample.pertoken = 1        



def sample_get_valid_start(start, dataset, warn):

    if start is None or not dataset.is_text_valid(start):
        new_start = dataset.get_random_vocab_item()
        if start is not None and warn:
            print(f"Text '{start}' includes tokens/chars not available in the dataset. Using random '{new_start}' instead")
        start = new_start

    return start



# -----------------------------------------------------------------------------
@torch.no_grad()
def sample(sample_config, model, train_dataset, stop_asap=None):
    """ stop_asap=[False] - when set to True, must break and return """

    eot_token = train_dataset.get_eot_token()

    model.eval()

    start = sample_get_valid_start(sample_config.start, train_dataset, True)
    ix = train_dataset.encode(start)
    x = torch.tensor(ix, dtype=torch.long).to(model.device)

    if sample_config.pertoken:

        def emit(idx):

            idx=idx[0].tolist()

            is_eot = idx[0] == eot_token

            if is_eot and sample_config.eotstop==-1:
                return -1

            chars = train_dataset.bufd_decode(idx)
            print(chars, sep='', end='', flush=True)

            if is_eot and sample_config.eotstop==1:
                return -1

            return 0


        x = x.repeat([1, 1])

        for t in range(sample_config.count):
            if t: print_sepline()

            print(start, sep='', end='')

            model.generate(x, sample_config.len, temperature=sample_config.temp, do_sample=True, top=sample_config.top, 
                           token_callback = emit if sample_config.pertoken else None,
                           stop_asap=stop_asap)

            if stop_asap is not None and stop_asap[0]:
                return

            # flush any buffered utf-8 characters
            chars = train_dataset.bufd_flush()
            print(chars, sep='', end='', flush=True)

            print()


    else:
        x = x.repeat([sample_config.count, 1])
        y = model.generate(x, sample_config.len, temperature=sample_config.temp, do_sample=True, top=sample_config.top, 
                           token_callback = emit if sample_config.pertoken else None,
                           stop_asap=stop_asap)

        if stop_asap is not None and stop_asap[0]:
            return

        for ir in range(y.size(0)):
            if ir: print_sepline()

            row = y[ir,:].tolist()

            if sample_config.eotstop:
              index = row.index(eot_token)
              if index >= 0:
                row = row[:index if sample_config.eotstop==-1 else index+1]

            completion = train_dataset.decode(row)

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


    sample_config = copy.copy(config.sample)
    sample_config.pertoken = 1


    stop_asap = [False]

    def signal_handler(signal, frame):
        nonlocal stop_asap
        print('\n<stopping>')
        stop_asap[0] = True

    original_sigint = signal.getsignal(signal.SIGINT)


    def print_help():
        print("Enter sampling start text or a command in the form -cmd or -cmd=val. Possible commands: ", ['-' + c for c in allowed_cmds], "\n Press Ctrl+C once to stop generation.")

    first = True
    while True:
        p = ''
        if sample_config.multiline:
            prompt='V\n'
        else:
            prompt='> '

        while True:
            try:
                p += input(prompt)
            except EOFError:
                break

            if not sample_config.multiline:
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
                    sample_config.merge_from_args(cmd_list, key_must_exist=True)

            if quit:
                break
        else:
            p = p.replace("\\n", "\n")
            sample_config.start = p

            stop_asap = [False]
            signal.signal(signal.SIGINT, signal_handler)

            sample(sample_config, model, train_dataset, stop_asap=stop_asap)

            signal.signal(signal.SIGINT, original_sigint)



