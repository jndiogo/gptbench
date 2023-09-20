"""

"""

import os, sys, copy, signal, json

import torch
import numpy as np

from .dataset import dataset_class_from_name, DATASET_CLASS_MAP
from .model import GPT
from .trainer import Trainer

from .config import empty_config, default_full_config, LogFlag, checkpoint_load, checkpoint_exists, dataset_get_default_config, dataset_checkpoint_config_keys

from .utils import CfgNode, print_sepline, set_seed



# -----------------------------------------------------------------------------
class Sample:

    @staticmethod
    def get_default_config():

        # sample.*
        c = CfgNode()

        c.count = 1 # number of generations
        c.max_len = 100 # max generated token count
        
        c.start_text = None # None: use random vocabulary item on each sampling. Or str with starting text
        c.start_after = None
        c.stop_before = None

        c.per_token = 1 # display each token immediately
        c.eot_stop = 0 # 0 don't stop, -1 stop before, 1 stop after (and display it)

        c.top = 0 # top_k/top_p  0: off,  ]0..1]: top_p,  [-1..0[: top_k(vocab_size * -top),  >=1: top_k(int(n))
        c.temp = 1. # temperature

        c.multiline = 0 # prompt mode: input multiple lines until a Ctrl+D or Ctrl+Z (in Windows)

        return c

    @staticmethod
    def checkpoint_config_keys():
        return ['count', 'max_len', 'start_text', 'per_token', 'eot_stop', 'top', 'temp', 'multiline']





    def __init__(self, name='model', work_dir='./out', log_mask=LogFlag.ALL):

        self.name = name
        self.work_dir = work_dir
        self.log_mask = log_mask

        self.config = default_full_config()

        self.model = None
        self.trainer = None

        self.train_dataset = None
        self.val_dataset = None

        self._model_path_prefix = os.path.join(self.work_dir, self.name)
        self._resumed_optimizer_state_dict = None
        self._can_train = False




    def set_datasets(self, class_name, train_path, val_path=None, train_split=None):

        assert class_name in DATASET_CLASS_MAP, f"Unknown dataset class '{class_name}'"
        assert (val_path is None) or (train_split is None), "Can't set both val_path and train_split"

        self.config.dataset.class_name = class_name
        self.config.dataset.train_path = train_path

        if val_path is not None:
            self.config.dataset.val_path_or_train_split = str(val_path)
        if train_split is not None:
            self.config.dataset.val_path_or_train_split = float(train_split)



    def init_new(self, over_config):
        self._init('new', over_config)

    def init_pretrained(self, init_type, over_config):
        self._check_pretrained_type(init_type)
        self._init(init_type, over_config)

    def resume(self, over_config):
        self._init('resume', over_config)


    def can_resume(self):
        return checkpoint_exists(self._model_path_prefix)



    def get_config(self):
        return copy.copy(self.config)








    @torch.no_grad()
    def prompt(self):
        """ """

        allowed_cmds = [
        'seed',
        'help',
        'quit',
        'config',

        'start_text',
        'count',
        'max_len',

        'per_token',
        'eot_stop',
        'top',
        'temp',
        'multiline',
        ]


        print("Prompt mode: press Enter (single line mode), or Ctrl+D / Ctrl+Z (multiline mode) to submit starting text. Enter -help for available commands.")


        sample_config = self.config.sample
        sample_config.per_token = 1 #  this is setting global config


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

                    if k == 'help' or k not in allowed_cmds:
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
                        print(self.config)

                    else:
                        cmd_list = [ '-' + k + '=' + v ]
                        sample_config.merge_from_args(cmd_list, key_must_exist=True)

                if quit:
                    break
            else:
                p = p.replace("\\n", "\n")
                sample_config.start_text = p

                stop_asap = [False]
                signal.signal(signal.SIGINT, signal_handler)

                self.sample(over_sample_config = sample_config, stop_asap=stop_asap)

                signal.signal(signal.SIGINT, original_sigint)





    # -----------------------------------------------------------------------------
    def _init(self, init_type, over_config):

        # set seed
        seed = over_config.get_or('seed', self.config.seed)
        set_seed(seed, self.log_mask & LogFlag.INIT)


        self._resumed_optimizer_state_dict = None

        # model init
        if init_type == 'new' or init_type == 'resume':

            if init_type == 'new':
                self.log(LogFlag.INIT, f"Initializing new model {self.name}")

                model_state_dict = None


            else: # load checkpoint
                from .train import Train

                self.log(LogFlag.INIT, f"Loading checkpoint from {self._model_path_prefix}")

                (model_state_dict, self._resumed_optimizer_state_dict, 

                 sample_config_dict,
                 train_config_dict,

                 model_config_dict,             
                 dataset_config_dict,
                 trainer_config_dict) = checkpoint_load(self._model_path_prefix, load_optimizer_state=self._can_train)
                # only load optimizer state if do_train

                # merge resumed configs into config
                self.config.sample.merge_from_dict(sample_config_dict, Sample.checkpoint_config_keys())
                self.config.train.merge_from_dict(train_config_dict, Train.checkpoint_config_keys())

                self.config.dataset.merge_from_dict(dataset_config_dict, dataset_checkpoint_config_keys())
                self.config.model.merge_from_dict(model_config_dict, GPT.checkpoint_config_keys())
                self.config.trainer.merge_from_dict(trainer_config_dict, Trainer.checkpoint_config_keys())

                #@ATTN - fix this:
                # if resumed dataset file is no longer available: erase it - either over_config's or an empty dummy will be used
                #if not os.path.isfile(config.dataset.train_path):
                #    config.dataset.train_path = None

                self.log(LogFlag.INIT, f"Checkpoint: iter_num={self.config.train.start_iter_num}, eval loss={self.config.train.start_eval_loss}", 2)

                self.config.train.start_iter_num += 1 # continue after last iteration



            # merge over_config into config for the final resolved config
            self.config.merge_from_config(over_config)

            # load datasets
            (self.train_dataset, self.val_dataset) = self._load_datasets()

            # ensure right vocab_size
            if init_type == 'resume':
                assert self.config.model.vocab_size == self.train_dataset.get_vocab_size(), f"Model vocab_size ({self.config.model.vocab_size} != Dataset vocab_size ({self.train_dataset.get_vocab_size()})"
            else:
                self.config.model.vocab_size = self.train_dataset.get_vocab_size()

            self.model = GPT(self.config.model)

            if model_state_dict is not None:
                self.model.load_state_dict( model_state_dict )


        elif init_type.startswith('gpt'):

            self.log(LogFlag.INIT, f"Initializing model from {init_type}")

            # merge over_config into config for the final resolved config
            self.config.merge_from_config(over_config)

            # will set config.model.* parameters as needed
            self.model, self.config.model = GPT.from_pretrained(init_type, self.config.model)

            # auto fill empty dataset as GPT2TokensDataset:
            if self.config.dataset.class_name is None:
                self.config.dataset.class_name = 'gpt2'

            # create a training dataset: possibly dummy with one sample
            (self.train_dataset, self.val_dataset) = self._load_datasets()

            # ensure right vocab_size
            self.config.model.vocab_size = self.train_dataset.get_vocab_size()



        self.log(LogFlag.INIT, f"Dataset: {self.config.dataset.train_path if self.config.dataset.train_path else 'dummy empty dataset'} vocab_size: {self.train_dataset.get_vocab_size()}")


        # model and dataset(s) are now loaded, settle/resolve config options
       
        # check sample.start
        if self.config.sample.start_text is not None:
            if not train_dataset.is_text_valid(self.config.sample.start_text):
                self.log(LogFlag.INIT, f"Config sample.start_text is not valid for dataset's vocabulary. Set to None (random)")
                self.config.sample.start_text = None # random vocab item on each sampling

        if self.config.sample.top > self.config.model.vocab_size:
            self.log(LogFlag.INIT, f'Config sample.top only up to vocab_size: {self.config.model.vocab_size}')
            self.config.sample.top = self.config.model.vocab_size






    def _load_datasets(self):

        dataset_config = self.config.dataset
        block_size = self.config.model.block_size

        assert block_size is not None, "Must set config.model.block_size"

        try:
            cls = dataset_class_from_name(dataset_config.class_name)
        except KeyError:
            assert False, f"Unknown config value dataset.class_name '{dataset_config.class_name}'"

        return cls.load_train_val_datasets(dataset_config.train_path,
                                           dataset_config.val_path_or_train_split,
                                       block_size,
                                       repeat_if_needed=True,
                                       verbose=self.log_mask & LogFlag.INIT)


    def _ensure_work_dir(self):
        # setup_work_dir: create the work directory if it doesn't already exist
        os.makedirs(self.work_dir, exist_ok=True)



    def _get_valid_start_text(self, start_text, dataset, warn):

        if start_text is None or not dataset.is_text_valid(start_text):
            new_start_text = dataset.get_random_vocab_item()
            if start_text is not None and warn:
                print(f"Text '{start_text}' includes tokens/chars not available in the dataset. Using random '{new_start_text}' instead")
            start_text = new_start_text

        return start_text


    def in_log(self, log_mask: LogFlag):
        return bool(log_mask & self.log_mask)

    def log(self, log_mask: LogFlag, *args, **kwargs):
        if self.in_log(log_mask):
            print(*args, **kwargs)



    def _check_pretrained_type(self, type):
        assert type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], 'init type must be one of: new, resume, gpt2, gpt2-medium, gpt2-large, gpt2-xl'







    @torch.no_grad()
    def sample(self, stop_asap=None, 
               over_sample_config=None, **over_sample_config_kwargs):

        """
        stop_asap=[False] - when set to True, sample will stop and return.

        over_sample_config: partial config to override config.sample settings.
        over_sample_config_kwargs: key values to override config.sample settings (and any over_sample_config).

        """

        #override existing keys
        if over_sample_config is not None:
            self.config.sample.merge_from_config(over_sample_config)
        self.config.sample.merge_from_dict(over_sample_config_kwargs, existing_only=True)



        sample_config = self.config.sample

        start_text = self._get_valid_start_text(sample_config.start_text, self.train_dataset, True)

        count = sample_config.count


        chars_buffer = [''] * count

        self._sample(None,
                     start_text, 
                     count, sample_config.max_len, 
                     sample_config.temp, sample_config.top, 
                     sample_config.eot_stop, sample_config.per_token,
                     stop_asap)






    @torch.no_grad()
    def _sample(self, 
                chars_callback,
                start_text, count, max_len, 
                temp, top,
                eot_stop, per_token,
                
                stop_asap=None):

        """
        stop_asap=[False] - when set to True, sample will stop and return.

        """

        sample_config = self.config.sample

        eot_token = self.train_dataset.get_eot_token()

        # parallel emitting along batch dim count
        chars_buffer = [''] * count
        emitting = [True] * count 
        emitted = [False] * count # any emission before?


        def emit_callback(idx, islast):
            # idx.shape=(count,1)
            idx=idx.numpy(force=True)

            nonlocal chars_buffer, emitting, emitted

            b=idx.shape[0]

            new_chars = self.train_dataset.bufd_decode(idx)

            for ib in range(b):

                id = idx[ib]

                if eot_stop==-1 and id == eot_token:
                    emitting[ib] = False
                    continue

                chars = new_chars[ib]

                if emitting:
                    if not emitted[ib]:
                        emitted[ib]=True
                        chars = start_text + chars

                    if per_token and len(chars):
                        print(chars, sep='', end='', flush=True)
                    else:
                        chars_buffer[ib] += chars


                    if eot_stop==1 and id == eot_token:
                        emitting[ib] = False
                        continue

            return all(emitted) and not any(emitting) # stop generating if...




        self.model.eval()

        ix = self.train_dataset.encode(start_text)
        x = torch.tensor(ix, dtype=torch.long).to(self.model.device)

        x = x.repeat([count, 1])

        y = self.model.generate(x, max_len, temperature=temp, do_sample=True, top=top, 
                                token_callback=emit_callback,
                                stop_asap=stop_asap)


        if not per_token: # print buffered

            for c in range(len(chars_buffer)):
                if c: print_sepline()

                print(chars_buffer[c])


        return y






    @torch.no_grad()
    def _sample_with_callback(self, 
                              start_text, count, max_len, 
                              temp, top, 
                              chars_callback=None, token_callback=None, 
                              stop_asap=None):
        """
        Sample into chars_callback(str-list, islast) or token_callback(numpy2D, islast).

        chars_callback can be called with empty strings in list
        
        Callbacks can return any non-None/zero value to stop sampling.
        """

        assert (chars_callback is None) ^ (token_callback is None), "Params chars_callback and token_callback are mutually exclusive"

        self.model.eval()

        ix = self.train_dataset.encode(start_text)
        x = torch.tensor(ix, dtype=torch.long).to(self.model.device)

        x = x.repeat([count, 1])

        def emit_callback(idx, islast): # only complete decoded chars/entities are sent here
            # idx.shape=(count,1)
            idx=idx.numpy(force=True)

            if token_callback is not None:
                should_stop = token_callback(idx, islast)
            else:
                chars = self.train_dataset.bufd_decode(idx)
                should_stop = callback(chars, islast=islast)

            return should_stop


        y = self.model.generate(x, max_len, temperature=temp, do_sample=True, top=top, 
                                token_callback=emit_callback,
                                stop_asap=stop_asap)

        return y




