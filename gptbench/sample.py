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


