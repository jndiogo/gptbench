"""

"""

import os, sys, copy, signal, json

import torch


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
        c.len = 100 # token count
        
        c.start = None # None: use random vocabulary item on each sampling. Or str with starting text
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
        return ['count', 'len', 'start', 'per_token', 'eot_stop', 'top', 'temp', 'multiline']





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
    def sample(self, over_sample_config=None, stop_asap=None, **kwargs):
        """ kwargs: key value of config.sample settings 
        stop_asap=[False] - when set to True, sample will stop and return """


        if over_sample_config is not None:
            self.config.sample.merge_from_config(over_sample_config)

        #override exsisting keys from kwargs
        self.config.sample.merge_from_dict(kwargs, existing_only=True)


        sample_config = self.config.sample

        eot_token = self.train_dataset.get_eot_token()

        self.model.eval()

        start = self._get_valid_start(sample_config.start, self.train_dataset, True)
        ix = self.train_dataset.encode(start)
        x = torch.tensor(ix, dtype=torch.long).to(self.model.device)

        if sample_config.per_token:

            def emit(idx):

                idx=idx[0].tolist()

                is_eot = idx[0] == eot_token

                if is_eot and sample_config.eot_stop==-1:
                    return -1

                chars = self.train_dataset.bufd_decode(idx)
                print(chars, sep='', end='', flush=True)

                if is_eot and sample_config.eot_stop==1:
                    return -1

                return 0


            x = x.repeat([1, 1])

            for t in range(sample_config.count):
                if t: print_sepline()

                print(start, sep='', end='')

                self.model.generate(x, sample_config.len, temperature=sample_config.temp, do_sample=True, top=sample_config.top, 
                               token_callback = emit if sample_config.per_token else None,
                               stop_asap=stop_asap)

                if stop_asap is not None and stop_asap[0]:
                    return

                # flush any buffered utf-8 characters
                chars = self.train_dataset.bufd_flush()
                print(chars, sep='', end='', flush=True)

                print()


        else:
            x = x.repeat([sample_config.count, 1])
            y = self.model.generate(x, sample_config.len, temperature=sample_config.temp, do_sample=True, top=sample_config.top, 
                               token_callback = emit if sample_config.per_token else None,
                               stop_asap=stop_asap)

            if stop_asap is not None and stop_asap[0]:
                return

            for ir in range(y.size(0)):
                if ir: print_sepline()

                row = y[ir,:].tolist()

                if sample_config.eot_stop:
                  index = row.index(eot_token)
                  if index >= 0:
                    row = row[:index if sample_config.eot_stop==-1 else index+1]

                completion = self.train_dataset.decode(row)

                print(completion)





    @torch.no_grad()
    def prompt(self, sample_config=None):
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
                sample_config.start = p

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
        if self.config.sample.start is not None:
            if not train_dataset.is_text_valid(self.config.sample.start):
                self.log(LogFlag.INIT, f"Config sample.start is not valid for dataset's vocabulary. Set to None (random)")
                self.config.sample.start = None # random vocab item on each sampling

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



    def _get_valid_start(self, start, dataset, warn):

        if start is None or not dataset.is_text_valid(start):
            new_start = dataset.get_random_vocab_item()
            if start is not None and warn:
                print(f"Text '{start}' includes tokens/chars not available in the dataset. Using random '{new_start}' instead")
            start = new_start

        return start


    def in_log(self, log_mask: LogFlag):
        return bool(log_mask & self.log_mask)

    def log(self, log_mask: LogFlag, *args, **kwargs):
        if self.in_log(log_mask):
            print(*args, **kwargs)



    def _check_pretrained_type(self, type):
        assert type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], 'init type must be one of: new, resume, gpt2, gpt2-medium, gpt2-large, gpt2-xl'




