"""

"""

import os, sys, copy, signal, json

import torch
import numpy as np

from .model import GPT
from .trainer import Trainer

from .config import empty_config, default_full_config, LogFlag, checkpoint_load, checkpoint_exists, dataset_get_default_config, dataset_checkpoint_config_keys, dataset_class_from_name, DATASET_CLASS_MAP

from .utils import CfgNode, print_sepline, set_seed, dict_from_str



# -----------------------------------------------------------------------------
DEFAULT_NAME = 'model'
DEFAULT_WORK_DIR = './models'
LOG_DIR = 'logs'

class Sample:

    @staticmethod
    def get_default_config():

        # sample.*
        c = CfgNode()

        c.max_len = 100 # max generated token count
        
        c.count = 1 # how many times to generate from the same start_text

        c.start_text = None # None: use random vocabulary item on each sampling. A str with starting text. If separated with start_text_sep multiple star_text are used (count is set to 1)
        c.start_after = None # when sampling, only emit after this text has been seen
        c.stop_before = None # when sampling, stop before emitting this. With flush=1 only works for single chars
        c.emit_start = 1 # on sampling, emit start_text? Only if start_after is None
        c.start_text_sep = '|' # when used in start_text, this char separates multiple start strings

        c.flush = 1 # display each token immediately
        c.eot_stop = 0 # 0 don't stop, -1 stop before, 1 stop after (and display it)

        c.top = 0 # top_k/top_p  0: off,  ]0..1]: top_p,  [-1..0[: top_k(vocab_size * -top),  >=1: top_k(int(n))
        c.temp = 1. # temperature

        c.multiline_prompt = 0 # prompt mode: input multiple lines until a Ctrl+D or Ctrl+Z (in Windows)

        return c

    @staticmethod
    def checkpoint_config_keys():
        return ['count', 'max_len', 'start_text', 'start_after', 'stop_before', 'emit_start', 'start_text_sep', 'flush', 'eot_stop', 'top', 'temp', 'multiline_prompt']





    def __init__(self, name=DEFAULT_NAME, work_dir=DEFAULT_WORK_DIR, log_mask=LogFlag.ALL):

        self.name = name
        self.log_mask = log_mask

        self.config = default_full_config()

        self.model = None
        self.trainer = None

        self.train_dataset = None
        self.val_dataset = None

        self.path = os.path.join(work_dir, self.name, '').replace(os.sep, '/')
        self.log_path = os.path.join(self.path, LOG_DIR, '').replace(os.sep, '/')

        self._resumed_optimizer_state_dict = None
        self._can_train = False




    def set_datasets(self, class_name, 
                     train_path, val_path=None, train_split=None,
                     params_str=None,
                     **params_kwargs):

        assert class_name in DATASET_CLASS_MAP, f"Unknown dataset class '{class_name}'"
        assert (val_path is None) or (train_split is None), "Can't set both val_path and train_split"
        assert (params_str is not None) ^ bool(len(params_kwargs)), "Only params_str or kwargs can be given"

        self.config.dataset.class_name = class_name
        self.config.dataset.train_path = train_path

        if val_path is not None:
            self.config.dataset.val_path_or_train_split = str(val_path)
        if train_split is not None:
            self.config.dataset.val_path_or_train_split = float(train_split)

        params=''
        if len(params_kwargs):
            for k,v in params_kwargs.items():
                if len(params): 
                    params = ',' + params
                params += k + '=' + str(v).replace(',' , '\,')
        elif params_str is not None:
            params = params_str

        if len(params):
            self.config.dataset.params = params



    def init_new(self, over_config):
        self._init('new', over_config)

    def init_pretrained(self, init_type, over_config):
        self._check_pretrained_type(init_type)
        self._init(init_type, over_config)

    def init_resume(self, over_config):
        self._init('resume', over_config)


    def can_resume(self):
        return checkpoint_exists(self.path)



    def get_config(self):
        return copy.copy(self.config)







    # -----------------------------------------------------------------------------
    def _init(self, init_type, over_config):

        # set seed
        seed = over_config.get_or('seed', self.config.seed)
        set_seed(seed, self.log_mask & LogFlag.INIT)

        assert self.config.dataset.has('class_name') and self.config.dataset.has('train_path'), "Need dataset to init. Set config.dataset: class_name and train_path. Or call set_datasets()"

        self._resumed_optimizer_state_dict = None

        # model init
        if init_type == 'new' or init_type == 'resume':

            if init_type == 'new':
                self.log(LogFlag.INIT, f"Initializing new model {self.name}")

                model_state_dict = None


            else: # load checkpoint
                from .train import Train

                self.log(LogFlag.INIT, f"Loading checkpoint from {self.path}")

                (model_state_dict, self._resumed_optimizer_state_dict, 

                 sample_config_dict,
                 train_config_dict,

                 model_config_dict,             
                 dataset_config_dict,
                 trainer_config_dict) = checkpoint_load(self.path, load_optimizer_state=self._can_train)
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


            # merge over_config into config for the final resolved config
            self.config.merge_from_config(over_config)

            # load datasets
            (self.train_dataset, self.val_dataset) = self._load_datasets()

            # ensure right vocab_size
            if init_type == 'resume':

                epoch = Trainer.calc_epoch_from_sample_num(self.config.train.sample_num,
                                                           len(self.train_dataset))
                iter_num = Trainer.iter_from_sample(self.config.train.sample_num, 
                                                    self.config.trainer.batch_size)
                self.log(LogFlag.INIT, f"Checkpoint: num={iter_num} ({epoch:.3f} epoch), loss train={self.config.train.train_loss:.4f} val={self.config.train.val_loss:.4f} eval->{self.config.train.eval_loss:.4f}")

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



        self.log(LogFlag.INIT, f"Dataset train_path: {self.config.dataset.train_path if self.config.dataset.train_path else 'dummy empty dataset'}, val_path_or_train_split: {self.config.dataset.val_path_or_train_split}, vocab_size: {self.train_dataset.get_vocab_size()}")


        # model and dataset(s) are now loaded, settle/resolve config options:
       
        # check sample.start
        if self.config.sample.start_text is not None:
            if not self.train_dataset.is_text_valid(self.config.sample.start_text):
                self.log(LogFlag.INIT, f"Config sample.start_text is not valid for dataset's vocabulary. Set to None for random start")
                self.config.sample.start_text = None # random vocab item on each sampling

        if self.config.sample.top > self.config.model.vocab_size:
            self.log(LogFlag.INIT, f'Config sample.top only up to vocab_size: {self.config.model.vocab_size}')
            self.config.sample.top = self.config.model.vocab_size

        if self.config.sample.start_text is not None:
            self.config.sample.start_text = self.config.sample.start_text.replace("\\n", "\n")
        if self.config.sample.start_after is not None:
            self.config.sample.start_after = self.config.sample.start_after.replace("\\n", "\n")
        if self.config.sample.stop_before is not None:
            self.config.sample.stop_before = self.config.sample.stop_before.replace("\\n", "\n")









    def update_sample_config(self, over_sample_config=None, **over_sample_config_kwargs):

        #override existing keys
        if over_sample_config is not None:
            self.config.sample.merge_from_config(over_sample_config)
        # override existing keys from kwargs
        self.config.sample.merge_from_dict(over_sample_config_kwargs, existing_only=True)





    @torch.no_grad()
    def sample(self, start_text, 
               dest='print',
               stop_asap=None, 
               **over_sample_config_kwargs):

        """
        stop_asap=[False] - when set to True, sample will stop and return.

        over_sample_config: partial config to override config.sample settings.
        over_sample_config_kwargs: key values to override config.sample settings (and any over_sample_config).

        """

        # save sample config so that any overrides are local here
        saved_sample_config = copy.copy(self.config.sample)

        sep = self.config.sample.start_text_sep

        if isinstance(start_text,list):
            sep.join(start_text)

        # override existing config.sample keys from kwargs
        self.config.sample.merge_from_dict(over_sample_config_kwargs, existing_only=True)

        sample_config = self.config.sample

        if start_text is None:
            start_text = self._get_valid_start_text(start_text, self.train_dataset, True)

        elif sep in start_text: # list of start texts
            start_text = start_text.split(sep)

            for i,text in enumerate(start_text):
                start_text[i] = self._get_valid_start_text(text, self.train_dataset, True)

        else:
            start_text = self._get_valid_start_text(start_text, self.train_dataset, True)


        def print_callback(strlist, islast):

            if sample_config.flush: # use the first line only
                if len(strlist[0]):
                    print(strlist[0], sep='', end='', flush=True)
                if islast:
                    print()

            else:
                for c in range(len(strlist)):
                    if c: print_sepline()
                    print(strlist[c])


        def list_callback(strlist, islast):
            nonlocal dest

            dest += strlist


        callback = list_callback if isinstance(dest, list) else print_callback

        if isinstance(start_text,list):
            # multiple start_text: no flush nor multiple count
            sample_config.count = 1
            sample_config.flush = 0

            for i,text in enumerate(start_text):
                if not isinstance(dest, list) and i: print_sepline()

                self.sample_callback(callback,
                                     sample_config.count, sample_config.max_len, 

                                     text, 
                                     start_after=sample_config.start_after,
                                     stop_before=sample_config.stop_before,
                                     emit_start=sample_config.emit_start,

                                     eot_stop=sample_config.eot_stop, flush=sample_config.flush,
                                     temp=sample_config.temp, top=sample_config.top,                              
                                     
                                     stop_asap=stop_asap)

        else:

            if sample_config.count > 1 or isinstance(dest, list):
                # count > 1: no flush (or big confusion)
                sample_config.flush = 0

            self.sample_callback(callback,
                                 sample_config.count, sample_config.max_len, 

                                 start_text, 
                                 start_after=sample_config.start_after,
                                 stop_before=sample_config.stop_before,
                                 emit_start=sample_config.emit_start,

                                 eot_stop=sample_config.eot_stop, flush=sample_config.flush,
                                 temp=sample_config.temp, top=sample_config.top,                              
                                 
                                 stop_asap=stop_asap)





        # restore saved config
        self.config.sample = saved_sample_config





    @torch.no_grad()
    def sample_callback(self, 
                        chars_callback,
                        count, max_len, 

                        start_text,
                        start_after=None,
                        stop_before=None,
                        emit_start=1,

                        eot_stop=0, flush=True,

                        temp=1.0, top=0.0,
                        
                        stop_asap=None):

        """
        Callback receives a list of str with sampled text. Some str in list may be ''.
        stop_asap=[False] - when set to True, sample will stop and return.

        """
        DEB = False

        # don't limit: if flush: count = 1

        if DEB: print(emit_start, start_after, stop_before)

        eot_token = self.train_dataset.get_eot_token()

        chars_buffer = [start_text if emit_start else ''] * count
        emitting = [start_after is None] * count 
        emitted = [False] * count # any emission before?


        def emit_callback(idx, islast):

            nonlocal chars_buffer, emitting, emitted

            # idx.shape=(count,1)
            idx=idx.numpy(force=True)

            b=idx.shape[0]

            new_chars_list = self.train_dataset.bufd_decode(idx)

            if DEB: print("pre", chars_buffer, new_chars_list, emitted[0], emitting[0])

            for ib in range(b):

                token_id = idx[ib]

                if eot_stop==-1 and token_id == eot_token:
                    emitting[ib] = False
                    continue

                # should start emitting?
                if not emitted[ib]: # never emitted, 
                    if start_after is not None:
                        # waiting for emit_after_text
                        acc_buffer = chars_buffer[ib] + new_chars_list[ib]

                        if (index := acc_buffer.find(start_after)) != -1: # start emitting!
                            new_chars_list[ib]=acc_buffer[index+1:]
                            chars_buffer[ib]=''
                            emitting[ib]=True

                    elif emit_start:
                        new_chars_list[ib]=chars_buffer[ib] + new_chars_list[ib]
                        chars_buffer[ib]=''
                        emitting[ib]=True

                    else:
                        emitting[ib]=True


                if DEB: print("mid", chars_buffer, new_chars_list, emitted[0], emitting[0])

                if emitting[ib]: # we're emitting

                    if not emitted[ib]: # not yet emitted
                        emitted[ib]=True

                    if stop_before is not None:
                        acc_buffer = chars_buffer[ib] + new_chars_list[ib]

                        if (index := acc_buffer.rfind(stop_before)) != -1:
                            chars_buffer[ib]=chars_buffer[ib][:index]
                            rem = index - len(acc_buffer) # rem is negative
                            new_chars_list[ib]=new_chars_list[ib][:rem]
                            emitting[ib] = False
                            continue

                    # accumulate
                    chars_buffer[ib] += new_chars_list[ib]

                    if eot_stop==1 and token_id == eot_token:
                        emitting[ib] = False
                        continue

                else:
                    new_chars_list[ib] = ''

            if DEB: print("post", chars_buffer, new_chars_list, emitted[0], emitting[0])

            if flush:
                chars_callback(new_chars_list, islast=islast)


            return all(emitted) and not any(emitting) # stop generating if...




        self.model.eval()

        ix = self.train_dataset.encode(start_text)
        x = torch.tensor(ix, dtype=torch.long).to(self.model.device)
        x = x.repeat([count, 1])

        y = self.model.generate(x, max_len, temperature=temp, do_sample=True, top=top, 
                                token_callback=emit_callback,
                                stop_asap=stop_asap)


        if (not flush) and not (stop_asap is not None and stop_asap[0]): # emit buffered - but not if stop_asap
            chars_callback(chars_buffer, islast=True)


        return y










    @torch.no_grad()
    def prompt(self,
               **over_sample_config_kwargs):

        """ """

        allowed_cmds = [
        'seed',
        'help',
        'quit',
        'config',

        'start_text',
        'count',
        'max_len',

        'flush',
        'eot_stop',
        'top',
        'temp',
        'multiline_prompt',
        ]


        print("Prompt mode: press Enter (single line mode), or Ctrl+D / Ctrl+Z (multiline mode) to submit starting text. Enter -help for available commands.")


        # save sample config so that any overrides are local to this function
        saved_sample_config = copy.copy(self.config.sample)


        # override existing config.sample keys from kwargs
        self.config.sample.merge_from_dict(over_sample_config_kwargs, existing_only=True)


        sample_config = self.config.sample
        sample_config.flush = 1 #  this is setting global config


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
            if sample_config.multiline_prompt:
                prompt='V\n'
            else:
                prompt='> '

            while True:
                try:
                    p += input(prompt)
                except EOFError:
                    break

                if not sample_config.multiline_prompt:
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
                start_text = p.replace("\\n", "\n")

                stop_asap = [False]
                signal.signal(signal.SIGINT, signal_handler)

                self.sample(start_text, over_sample_config=sample_config, stop_asap=stop_asap)

                signal.signal(signal.SIGINT, original_sigint)

                print()



        # restore saved config
        self.config.sample = saved_sample_config








    def _load_datasets(self):

        dataset_config = self.config.dataset
        block_size = self.config.model.block_size

        assert block_size is not None, "Must set config.model.block_size"

        try:
            cls = dataset_class_from_name(dataset_config.class_name)
        except KeyError:
            assert False, f"Unknown config value dataset.class_name '{dataset_config.class_name}'"

        # extra params?
        if dataset_config.params is not None:
            params = dataset_config.params.replace('\\n', '\n')
            kwargs = dict_from_str(params)
        else:
            kwargs = {}

        return cls.load_train_val_datasets(dataset_config.train_path,
                                           dataset_config.val_path_or_train_split,
                                           block_size,
                                           repeat_if_needed=True,
                                           verbose=self.log_mask & LogFlag.INIT,
                                           **kwargs)




    def _get_valid_start_text(self, start_text, dataset, warn):

        if start_text is None or not dataset.is_text_valid(start_text):
            new_start_text = dataset.get_random_vocab_item()
            if start_text is not None and warn:
                self.log(LogFlag.SAMPLE, f"Text '{start_text}' includes tokens/chars not available in the dataset. Using random '{new_start_text}' instead")
            start_text = new_start_text

        return start_text



    def _check_pretrained_type(self, type):
        assert type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], 'init type must be one of: new, resume, gpt2, gpt2-medium, gpt2-large, gpt2-xl'


    def ensure_path(self):
        # setup_path: create the work directory if it doesn't already exist
        #os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + LOG_DIR, exist_ok=True)


    def path_save(self, filename, text):
        with open(os.path.join(self.path, filename), 'w', encoding='utf-8') as f:
            f.write(text)


    def in_log(self, log_mask: LogFlag):
        return bool(log_mask & self.log_mask)

    def log(self, log_mask: LogFlag, *args, **kwargs):
        if self.in_log(log_mask):
            print(*args, **kwargs)


