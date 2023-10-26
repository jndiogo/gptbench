"""
The Sample class can do model inference. For training (and sampling) see the Train class.
"""

import os, sys, copy, signal, json, math

import torch
import numpy as np

from .model import GPT
from .trainer import Trainer

from .config import LogFlag, empty_config, full_default_config, checkpoint_load, checkpoint_exists, dataset_get_default_config, dataset_class_from_name, DATASET_CLASS_MAP

from .conf import Conf
from .utils import print_sepline, set_all_random_seeds, str_dict_from_str



# -----------------------------------------------------------------------------
DEFAULT_NAME = 'model'
DEFAULT_WORK_DIR = './checkpoints'
LOG_SUBDIR = 'logs'

class Sample:

    @staticmethod
    def get_default_config():

        # sample.*
        c =Conf()

        c.setup('max_len', 100, int, 'Max generated token count')
        
        c.setup('count', 1, int, 'How many times to generate from the same start_text')


        c.setup('start_text', None, str, 'Starting text for generation. None: use random vocabulary item on each sampling. A str with starting text.If separated with start_text_sep multiple star_text are used (count is set to 1)')

        c.setup('start_text_sep', '|', str, 'When used in start_text, this char separates multiple start strings')


        c.setup('emit_start', True, bool, 'When sampling, emit start_text? Only if emit_after is None')
        
        c.setup('emit_after', None, str, 'When sampling, only emit after this text has been seen')
        
        c.setup('emit_before', None, str, 'When sampling, stop before emitting this. With flush=1 only works for single chars')

        
        c.setup('flush', True, bool, 'When sampling, should each token display immediately?')

        c.setup('eot_stop', 0, int, "Should generation stop when dataset's special End-Of-Text token is emitted? 0=don't stop, -1=stop before, 1=stop after (and display it)")

        c.setup('top', 0., float, 'Top_k or top_p filtering: 0: off,  ]0..1]: top_p,  [-1..0[: top_k(vocab_size * -top),  >=1: top_k(int(n))')
        c.setup('temp',  1., float, "Temperature")


        c.setup('max_batch_size', None, int, "Maximum batch size when inferring in parallel with multiple start text. None means same as trainer.batch_size config entry")

        c.setup('multiline_prompt', False, bool, 'On prompt mode: input multiple lines until a Ctrl+D or Ctrl+Z (in Windows)')

        return c



    def __init__(self, name=DEFAULT_NAME, work_dir=DEFAULT_WORK_DIR, 
                 seed=None, 
                 log_mask=LogFlag.COMMON):

        self.work_dir = work_dir
        self.log_mask = log_mask

        self.reset(name)

        self._can_train = False
        self.trainer = None

        if seed is not None:
            self.set_seed(seed)




    def init_new(self, over_config, name=None):
        self._init('new', over_config, name)

    def init_pretrained(self, init_type, over_config=None, name=None):
        self._check_pretrained_type(init_type)
        self._init(init_type, over_config, name)


    def load(self, over_config=None, name=None):
        self._init('resume', over_config, name)

    def can_load(self, name=None):
        if name is not None:
            path = os.path.join(self.work_dir, name, '').replace(os.sep, '/')
        else:
            path = self.path

        return checkpoint_exists(path)



    def reset(self, name=None, hard_reset=True):
        if name is None:
            name=DEFAULT_NAME
        self.set_name(name)

    
        self.state = { 'n_samples': 0, # number of trained samples so far
                       'train_loss': float('inf'), # last evaluated train dataset loss 
                       'val_loss': float('inf'), # last evaluated validation dataset loss
                       'eval_loss': float('inf') # last evaluation loss calculated from train_loss and val_loss according to eval_type
                       }

        self.last_saved_state = copy.copy(self.state)
        
        self.model = None
        self.trainer = None

        self._loaded_optimizer_state_dict = None

        if hard_reset:
            self.config = full_default_config()

            self.train_dataset = None
            self.val_dataset = None



    def set_datasets(self, 
                     # either call with one or two initialized datasets with same block_size as model.block_size:
                     train_dataset=None, val_dataset=None,
                     
                     # or have GPTBench create it by passing:
                     class_name=None, train_path=None, train_split=None, val_path=None, 
                     params_str=None, **params_kwargs
                     ):

        """ Set already constructed datasets or config for later creation """

        assert (train_dataset is not None) ^ (class_name is not None), "Either pass a constructed dataset by setting train_dataset/val_dataset or pass a class_name, train_path to be constructed later"

        if train_dataset is not None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

            self.config.dataset.class_name = self.config.dataset.train_path = self.config.dataset.val_path = self.config.dataset.params = None

        else:
            assert class_name in DATASET_CLASS_MAP, f"Unknown dataset class '{class_name}'"
            assert not ((params_str is not None) and bool(len(params_kwargs))), "Only one of params_str or **params_kwargs can be given"

            self.train_dataset = self.val_dataset = None

            self.config.dataset.class_name = class_name

            self.config.dataset.train_path = train_path
            self.config.dataset.train_split = train_split

            self.config.dataset.val_path = val_path

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




    # -----------------------------------------------------------------------------
    def _init(self, init_type, over_config=None, name=None):

        self.reset(name if name is not None else self.name, hard_reset=False)

        if over_config is None:
            over_config = empty_config()

        if name is not None:
            self.set_name(name)


        # set seed - gather setting from full and overriding config
        seed = over_config.get('seed', self.config.seed)
        if seed != -1:
            seed = self.set_seed(seed)
            self.log(LogFlag.INIT, f"Set random seed: {seed}")


        assert self.config.dataset.has('class_name') and self.config.dataset.has('train_path'), "Need a dataset to init. Set config.dataset with class_name and train_path. Or call set_datasets()"


        self._loaded_optimizer_state_dict = None

        # model init
        if init_type == 'new' or init_type == 'resume':

            if init_type == 'new':
                self.log(LogFlag.INIT, f"Initializing new model {self.name}")

                model_state_dict = None


            else: # load checkpoint
                from .train import Train

                self.log(LogFlag.INIT, f"Loading checkpoint from {self.path}")

                (state_dict, config_dict,
                model_state_dict, self._loaded_optimizer_state_dict) = checkpoint_load(self.path, load_optimizer_state=self._can_train)

                # update state and config resumeds
                self.state.update(state_dict)
                self.last_saved_state = copy.copy(self.state)

                self.config.update(config_dict)


                #@ATTN - fix this:
                # if resumed dataset file is no longer available: erase it - either over_config's or an empty dummy will be used
                #if not os.path.isfile(config.dataset.train_path):
                #    config.dataset.train_path = None


            # finally update global config from users's over_config, for the final resolved config
            self.config.update(over_config)
    
            # load datasets
            (self.train_dataset, self.val_dataset) = self._load_datasets()

            # ensure right vocab_size
            if init_type == 'resume':

                epoch = Trainer.calc_epoch_from_sample_num(self.state['n_samples'],
                                                           len(self.train_dataset))
                iter_num = Trainer.iter_from_sample(self.state['n_samples'], 
                                                    self.config.trainer.batch_size)
                self.log(LogFlag.INIT, f"Checkpoint: iter={iter_num} ({epoch:.3f} epoch), loss train={self.state['train_loss']:.4f} val={self.state['val_loss']:.4f} eval->{self.state['eval_loss']:.4f}")

                assert self.config.model.vocab_size == self.train_dataset.get_vocab_size(), f"Model vocab_size ({self.config.model.vocab_size}) != Dataset vocab_size ({self.train_dataset.get_vocab_size()})"

            else:
                self.config.model.vocab_size = self.train_dataset.get_vocab_size()


            self.model = GPT(self.config.model)

            if model_state_dict is not None:
                self.model.load_state_dict( model_state_dict )


        elif init_type.startswith('gpt'):

            self.log(LogFlag.INIT, f"Initializing model from {init_type}")

            # update from over_config
            self.config.update(over_config)

            # will set config.model.* parameters as needed
            self.model, self.config.model = GPT.from_pretrained(init_type, self.config.model)

            # auto fill empty dataset as GPT2TokensDataset:
            if self.config.dataset.class_name is None:
                self.config.dataset.class_name = 'gpt2'

            # create a training dataset: possibly dummy with one sample
            (self.train_dataset, self.val_dataset) = self._load_datasets()

            # ensure right vocab_size
            self.config.model.vocab_size = self.train_dataset.get_vocab_size()



        self.log(LogFlag.INIT, f"Dataset train_path: {self.config.dataset.train_path if self.config.dataset.train_path else 'dummy empty dataset'}, val_path: {self.config.dataset.val_path}, train_split: {self.config.dataset.train_split}, vocab_size: {self.train_dataset.get_vocab_size()}")

        self.log(LogFlag.INIT, f"Model params: {self.model.get_num_params() * 1e-6:.2f}M")


        # model and dataset(s) are now loaded, settle/resolve config options:
        if self.config.sample.start_text is not None:
            if not self.train_dataset.is_text_valid(self.config.sample.start_text):
                self.log(LogFlag.INIT, f"Config sample.start_text is not valid for dataset's vocabulary. Set to None for random start")
                self.config.sample.start_text = None # random vocab item on each sampling

        if self.config.sample.top > self.config.model.vocab_size:
            self.log(LogFlag.INIT, f'Config sample.top only up to vocab_size: {self.config.model.vocab_size}')
            self.config.sample.top = self.config.model.vocab_size

        if self.config.sample.start_text is not None:
            self.config.sample.start_text = self.config.sample.start_text.replace("\\n", "\n")
        if self.config.sample.emit_after is not None:
            self.config.sample.emit_after = self.config.sample.emit_after.replace("\\n", "\n")
        if self.config.sample.emit_before is not None:
            self.config.sample.emit_before = self.config.sample.emit_before.replace("\\n", "\n")







    def update_sample_config(self, over_sample_config=None, **over_sample_config_kwargs):

        #override existing keys
        if over_sample_config is not None:
            self.config.sample.update(over_sample_config)
        # override existing keys from kwargs
        self.config.sample.update(over_sample_config_kwargs)





    @torch.no_grad()
    def sample(self, start_text, 
               dest='print',
               stop_asap=None, 
               **over_sample_config_kwargs):

        """
        stop_asap=[False] - when set to True, sample will stop and return.

        over_sample_config_kwargs: key values to override config.sample settings (and any over_sample_config).

        """

        # save sample config so that any overrides are local here
        saved_sample_config = copy.copy(self.config.sample)

        sep = self.config.sample.start_text_sep

        if isinstance(start_text,list):
            sep.join(start_text)

        # override existing config.sample keys from kwargs
        self.config.sample.update(over_sample_config_kwargs)

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


        if isinstance(start_text,list):
            # config's count and flush are ignored in sample_group_list, but print_callback reacts to flush, so:
            sample_config.flush = 0

            gen_list = self.sample_group_list(sample_config.max_len, 

                                              start_text, 
                                              emit_after=sample_config.emit_after,
                                              emit_before=sample_config.emit_before,
                                              emit_start=sample_config.emit_start,

                                              eot_stop=sample_config.eot_stop,
                                              temp=sample_config.temp, top=sample_config.top,

                                              max_batch_size=self._get_sample_max_batch_size(),
                                            
                                              stop_asap=stop_asap)

            if isinstance(dest, list):
                dest[:] = gen_list
            else:
                print_callback(gen_list,True)

        else:


            callback = list_callback if isinstance(dest, list) else print_callback


            if sample_config.count > 1 or isinstance(dest, list):
                # count > 1: no flush (or big confusion)
                sample_config.flush = 0

            self.sample_callback(callback,
                                 sample_config.count, sample_config.max_len, 

                                 start_text, 
                                 emit_after=sample_config.emit_after,
                                 emit_before=sample_config.emit_before,
                                 emit_start=sample_config.emit_start,

                                 eot_stop=sample_config.eot_stop, flush=sample_config.flush,
                                 temp=sample_config.temp, top=sample_config.top,                              
                                 
                                 stop_asap=stop_asap)


        # restore saved config
        self.config.sample = saved_sample_config







    @torch.no_grad()
    def prompt(self,
               **over_sample_config_kwargs):

        """ """

        allowed_cmds = [
        'help',
        'quit',
        'config'
        'seed',
        ]

        allowed_cmds += Sample.get_default_config().keys()


        print("Prompt mode: press Enter (single line mode), or Ctrl+D / Ctrl+Z (multiline mode) to submit starting text. Enter -help for available commands.")


        # save sample config so that any overrides are local to this function
        saved_sample_config = copy.copy(self.config.sample)


        # override existing config.sample keys from kwargs
        self.config.sample.update(over_sample_config_kwargs)


        sample_config = self.config.sample
        sample_config.flush = True #  this is setting global config
        del sample_config.start_text # we'll provide this directly


        stop_asap = [False]

        def signal_handler(signal, frame):
            nonlocal stop_asap
            print('\n<stopping>')
            stop_asap[0] = True

        original_sigint = signal.getsignal(signal.SIGINT)


        def print_help():
            print("Help: Enter sampling start text or a command in the form -cmd or -cmd=val.")
            print("Possible commands:", ', '.join(["'-" + c + "'" for c in allowed_cmds]), "\n")
            print("=== Commands, values and help (prefix with '-' to use, ex: -cmd=1) ==================")
            Sample.get_default_config().help()
            print("=====================================================================================\n")
            print("Press Ctrl+C once to stop generation.")


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
                        self.set_seed(int(v))

                    elif k == 'config':
                        print("Config:")
                        print(self.config.dump(int(v)))

                    else:
                        cmd_list = [ '-' + k + '=' + v ]
                        sample_config.update_from_args(cmd_list)

                if quit:
                    break
            else:
                start_text = p.replace("\\n", "\n")

                stop_asap = [False]
                signal.signal(signal.SIGINT, signal_handler)

                self.sample(start_text, stop_asap=stop_asap, **sample_config.to_dict())

                signal.signal(signal.SIGINT, original_sigint)

                print()



        # restore saved config
        self.config.sample = saved_sample_config









    @torch.no_grad()
    def sample_group_list(self, 
                          max_len, 

                          start_text_list,
                          emit_after=None,
                          emit_before=None,
                          emit_start=True,

                          eot_stop=0,

                          temp=1.0, top=0.0,

                          max_batch_size=None,
                           
                          stop_asap=None):

        """ Split start_text_list into same sized batches, generate each one then asseble back in order """

        assert isinstance(start_text_list,list), "start_text_list must be a list of strings"

        # index,text to later restore order
        txl = [[i,s] for i,s in enumerate(start_text_list)]
        
        # split into same sized buckets
        ssb = {}
        for t in txl:
            lt = len(self.train_dataset.encode(t[1])) # must be the length of the encoded tokens: we're grouping into same-sized tensors
            if lt not in ssb:
                ssb[lt]=[]
            ssb[lt].append(t)


        def list_callback(strlist, islast):
            nonlocal dest
            dest += strlist

        max_batch_size = max_batch_size if max_batch_size is not None else int(1e10)

        for l,lst in ssb.items():

            start_lst = [t[1] for t in lst]

            # split in max_batch_sizes
            base_index = 0            
            while len(start_lst):
                part_lst = start_lst[:max_batch_size]
                start_lst = start_lst[max_batch_size:]

                dest=[]

                self.sample_callback(list_callback,
                                     1, max_len, 

                                     part_lst, 
                                     emit_after=emit_after,
                                     emit_before=emit_before,
                                     emit_start=emit_start,

                                     eot_stop=eot_stop, flush=False,
                                     temp=temp, top=top,                              
                                     
                                     stop_asap=stop_asap)

                for i, out in enumerate(dest):
                    lst[base_index + i].append(out)

                base_index += len(part_lst)

        #print(ssb)

        # reorder by indexes to return in the right order
        out = [''] * len(start_text_list)

        for lst in ssb.values():
            for s in lst:
                out[s[0]] = s[2]

        return out








    @torch.no_grad()
    def sample_callback(self, 
                        chars_callback,
                        repeat_count, max_len, 

                        start_text,
                        emit_after=None,
                        emit_before=None,
                        emit_start=1,

                        eot_stop=0, flush=True,

                        temp=1.0, top=0.0,
                        
                        stop_asap=None):

        """
        start_text can be a string or a list. If start_text is a list, all items must have equal dataset.encoded lengths. This is taken care of in sample_group_list(), if you need that.

        Callback receives a list of str with sampled text. Some str in list may be ''.
        stop_asap=[False] - when set to True, sample will stop and return.

        """
        DEB = False

        # don't limit: if flush: count = 1

        self.train_dataset.bufd_decode_init() # important to init before using as there might be stuck characters from previous bad unicode points

        eot_token = self.train_dataset.get_eot_token()

        if isinstance(start_text,list):
            repeat_count = 1
        else:
            assert isinstance(start_text,str), "start_text must be an str or str list"
            start_text=[start_text] * repeat_count

        text_count = len(start_text)

        if emit_start:
            chars_buffer = start_text
        else:
            chars_buffer = [''] * text_count

        emitting = [emit_after is None] * text_count
        emitted = [False] * text_count # any emission before?

        if DEB: print(start_text, text_count, repeat_count, emit_start, emit_after, emit_before)


        def emit_callback(idx, islast):

            nonlocal chars_buffer, emitting, emitted

            # idx.shape=(text_count,1)
            idx=idx.numpy(force=True)

            b=idx.shape[0]

            new_chars_list = self.train_dataset.bufd_decode(idx)

            if DEB: print("pre", chars_buffer, idx, new_chars_list, emitted[0], emitting[0])

            for ib in range(b):

                token_id = idx[ib]

                if eot_stop==-1 and token_id == eot_token:
                    emitting[ib] = False
                    continue

                # should start emitting?
                if not emitted[ib]: # never emitted, 
                    if emit_after is not None:
                        # waiting for emit_after_text
                        acc_buffer = chars_buffer[ib] + new_chars_list[ib]

                        if (index := acc_buffer.find(emit_after)) != -1: # start emitting!
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

                    if emit_before is not None:
                        acc_buffer = chars_buffer[ib] + new_chars_list[ib]

                        if (index := acc_buffer.rfind(emit_before)) != -1:
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

        coded_len = len(self.train_dataset.encode(start_text[0]))
        ix = torch.empty((text_count,coded_len), dtype=torch.long)

        for i,txt in enumerate(start_text):
            ix[i] = torch.tensor(self.train_dataset.encode(txt), dtype=torch.long)

        ix = ix.to(self.model.device)


        """
        ix = self.train_dataset.encode(start_text)
        x = torch.tensor(ix, dtype=torch.long).to(self.model.device)
        x = x.repeat([count, 1])
        """

        y = self.model.generate(ix, max_len, temperature=temp, do_sample=True, top=top, 
                                token_callback=emit_callback,
                                stop_asap=stop_asap)


        if (not flush) and not (stop_asap is not None and stop_asap[0]): # emit buffered - but not if stop_asap
            chars_callback(chars_buffer, islast=True)

        if DEB: print(y)

        return y






    # ----------------------------------------------------------------------------- Measure and estimate

    @torch.no_grad()
    def estimate_loss(self, train_dataset, val_dataset, batch_size, iters):
        """
        train_dataset or val_dataset can be None to skip its eval.
        Returns train_loss,val_loss any of which can be None.
        """

        self.model.eval()

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

                x, y = x.to(self.model.device), y.to(self.model.device)

                _, loss = self.model(x,y)

                losses[k] = loss.item()

            out.append(losses.mean().item())

        return out



    @torch.no_grad()
    def measure_qa(self, questions, answers, 

                    test_fn=None,                
                    log_list=None, log_cond=None,

                    stop_asap=None, 
                    **over_sample_config_kwargs):

        """
        test_fn: None means case sensitive string compare, returning 1. or 0.

        log_list: a list where entries that satisfy log_cond are logged. Tupple of (question, answer, generated)
        log_cond: >=0 log test results >= log_cond. <0: log test results <= -log_cond

        stop_asap=[False] - when set to True, sample will stop and return.

        over_sample_config_kwargs: key values to override config.sample settings (and any over_sample_config).

        """

        assert isinstance(questions, list), "questions must be a list"
        if answers is not None:
            assert isinstance(answers, list) and len(questions) == len(answers), "answers must be a list and same size as questions"


        # save sample config so that any overrides are local here
        saved_sample_config = copy.copy(self.config.sample)

        # override existing config.sample keys from kwargs
        self.config.sample.update(over_sample_config_kwargs)

        sample_config = self.config.sample

        gen_list = self.sample_group_list(sample_config.max_len, 

                                          questions, 
                                          emit_after=sample_config.emit_after,
                                          emit_before=sample_config.emit_before,
                                          emit_start=False,

                                          eot_stop=sample_config.eot_stop,
                                          temp=sample_config.temp, top=sample_config.top,

                                          max_batch_size=sample_config.max_batch_size,
                                        
                                          stop_asap=stop_asap)

        if stop_asap is not None and stop_asap[0]: # cancelled
            self.config.sample = saved_sample_config
            return None

        #print(gen_list)

        results = []

        if test_fn is None:
            def case_sensitive_equal(_question, answer, gen):
                return float(str(answer) == str(gen))

            assert answers is not None, "To use default test_fn please provide the answers list"

            test_fn = case_sensitive_equal


        for i in range(len(questions)):
            t = (questions[i],
                 None if answers is None else answers[i],
                 gen_list[i])

            res = test_fn(*t)

            # logging
            if log_list is not None:
                if log_cond >= 0:
                    if res >= log_cond:
                        log_list.append( t )
                elif res <= -log_cond:
                    log_list.append( t )

            results.append(res)

        # restore saved config
        self.config.sample = saved_sample_config

        return results




    def measure_accuracy(self, questions, answers, 
                         test_fn=None, # None means case sensitive string compare, returning 1. or 0.
                         log_list=None, log_cond=None,
                         stop_asap=None, 
                         **over_sample_config_kwargs):
        """
        log_list: a list where entries that satisfy log_cond are logged. Tupple of (question, answer, generated)
        log_cond: >=0 log test results >= log_cond. <0: log test results <= -log_cond
        """
        results = self.measure_qa(questions, answers, test_fn, log_list, log_cond, stop_asap, **over_sample_config_kwargs)

        return sum(results) / len(results)




    @torch.no_grad()
    def measure_loss(self, dataset, stride=-1, max_batch_size=None):
        """
        stride: measurements are done at block_size blocks. stride controls how to advance along the dataset at each evaluation. if > 0: sample position increment, 0..-1: -ratio of block size, for example -1 means increment a block_size on each evaluation.
        """

        assert self.model is not None, "No model set"

        block_size = self.model.block_size

        if stride < 0:
            stride = int(-stride * block_size)
        stride = max(1, stride)

        if max_batch_size is None:
            max_batch_size=self._get_sample_max_batch_size()

        self.model.eval()

        data_len = len(dataset)

        loss_list = []
        last_end = 0
        batch_x=[]
        batch_y=[]

        for begin in range(0, data_len, stride):

            end = min(begin + block_size, data_len)

            usable = end - last_end # don't account for mulitple losses for the same position, if stride < block_size

            xy = dataset[begin] # returns block_size tensors

            # block positions which were already accounted before
            xy[1][:-usable] = -1

            batch_x.append(xy[0])
            batch_y.append(xy[1])
            last_end = end


            if len(batch_x) == max_batch_size or end == data_len: # forward batch

                xb = torch.stack(batch_x).to(self.model.device)
                yb = torch.stack(batch_y).to(self.model.device)
                #print(xb.shape)

                batch_x.clear()
                batch_y.clear()

                _,loss = self.model(xb,yb) # loss cross_entropy was only averaged over valid y positions

                loss_list.append(loss.item())

                del _,loss, xb,yb



        # return average loss
        return sum(loss_list)/len(loss_list)





    def measure_perplexity(self, dataset, stride=-1, max_batch_size=None):

        loss = self.measure_loss(dataset, stride=stride, max_batch_size=max_batch_size)

        return math.exp(loss)




    @torch.no_grad()
    def model_forward(self, text):
        """
        No config.sample settings are used here.
        Returns logits,loss.
        """

        idx = self.train_dataset.encode(text)

        assert len(idx) <= self.model.block_size, f"Can only forward up to model.block_size {self.model.block_size} tokens, text has {len(idx)}"

        x = torch.tensor(idx, dtype=torch.long).to(self.model.device)
        y = torch.empty(x.shape, dtype=torch.long).to(self.model.device)
        y[:-1]=torch.as_tensor(idx[1:])
        y[-1]=-1
        
        return self.model(x.unsqueeze(0),y.unsqueeze(0))


    @torch.no_grad()
    def model_forward_argmax(self, text):
        """
        No config.sample settings are used here.
        Returns argmax,decoded text from argmax(logits)
        """

        logits,_ = self.model_forward(text)
        am = logits.argmax(dim=-1)
        return am, self.train_dataset.decode(am[0].tolist())




# -----------------------------------------------------------------------------

    def get_config(self):
        return copy.deepcopy(self.config)

    def get_trained_sample_count(self):
        return self.state['n_samples']

    def get_trained_iter_count(self):
        """ Depends on current batch_size. Might have been trained with other batch sizes before """
        return Trainer.iter_from_sample(self.state['n_samples'], 
                                        self.config.trainer.batch_size)


    def set_name(self, name):
        self.name = name
        self.path = os.path.join(self.work_dir, self.name, '').replace(os.sep, '/')
        self.log_path = os.path.join(self.path, LOG_SUBDIR, '').replace(os.sep, '/')


    def set_seed(self, seed):
        return set_all_random_seeds(seed)


    def path_append(self, filename, text, clear=False):
        with open(os.path.join(self.path, filename), 'w' if clear else 'a', encoding='utf-8') as f:
            f.write(text)


    def in_log(self, log_mask: LogFlag):
        return bool(log_mask & self.log_mask)

    def log(self, log_mask: LogFlag, *args, **kwargs):
        if self.in_log(log_mask):
            print(*args, **kwargs)


    def ensure_path(self):
        """ Create work and log directories if not already existing """
        os.makedirs(self.path + LOG_SUBDIR, exist_ok=True)








    def _load_datasets(self):

        dataset_config = self.config.dataset

        if dataset_config.class_name is not None:

            block_size = self.config.model.block_size

            assert block_size is not None, "Must set config.model.block_size"

            try:
                cls = dataset_class_from_name(dataset_config.class_name)
            except KeyError:
                assert False, f"Unknown config value dataset.class_name '{dataset_config.class_name}'"


            if dataset_config.train_path is not None:
                dataset_config.train_path = dataset_config.train_path.replace(os.sep, '/')
            if dataset_config.val_path is not None:
                dataset_config.val_path = dataset_config.val_path.replace(os.sep, '/')

            # extra params?
            if dataset_config.params is not None:
                params = dataset_config.params = dataset_config.params.replace('\\n', '\n')
                kwargs = str_dict_from_str(params)
            else:
                kwargs = {}


            # normalize path to forward slashes
            return cls.load_train_val_datasets(block_size,
                                               dataset_config.train_path, dataset_config.train_split,
                                               dataset_config.val_path,
                                               
                                               repeat_if_needed=True,
                                               verbose=self.log_mask & LogFlag.INIT,
                                               **kwargs)


        elif self.train_dataset is not None: # already created
            return self.train_dataset, self.val_dataset

        else:
            raise RuntimeError("Dataset must be defined in config's dataset.* or by calling set_dataset()")


    def _get_valid_start_text(self, start_text, dataset, warn):

        if start_text is None or not dataset.is_text_valid(start_text):
            new_start_text = dataset.get_random_vocab_item()
            if start_text is not None and warn:
                self.log(LogFlag.SAMPLE, f"Text '{start_text}' includes tokens/chars not available in the dataset. Using random '{new_start_text}' instead")
            start_text = new_start_text

        return start_text



    def _check_pretrained_type(self, type):
        assert type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], 'init type must be one of: new, resume, gpt2, gpt2-medium, gpt2-large, gpt2-xl'



    def _get_sample_max_batch_size(self):
        if self.config.sample.max_batch_size is not None:
            return self.config.sample.max_batch_size
        else:
            assert self.config.trainer.batch_size is not None, "config.trainer.batch_size cannot be None"
            return self.config.trainer.batch_size








    @staticmethod
    def estimate_max_batch_size(model_config, optimizer_type,
                                starting_size=512, delta_size=0.5, times=2):

        """
        Try allocating model/optimizer with decreasing batch_size until it fits memory.
        Should be called with empty memory before Sample or Train objects are created.
        Measuring memory is messy: on Jupyter notebooks GPU memory gets stuck if an exception goes uncatched.

        optimizer_type: string as in trainer.optimizer config setting or None for inference only.
        delta_size: >0: ratio for next size, <0: add to next size
        times should be >1 to simulate real conditions, like inside measure_loss() method. Possibly because memory freeing is asynchronous, freeing and then allocating will happen with previous memory still occupied?
        """

        print('Creating model')        
        model = GPT(model_config)

        if optimizer_type is not None:
            print(f'Creating optimizer {optimizer_type}')
            trainer_config = Trainer.get_default_config()
            trainer_config.optimizer = optimizer_type
            optimizer = model.configure_optimizers(trainer_config)
            model.train()
        else:
            optimizer = None
            model.eval()


        import gc
        def cleanup():
            gc.collect()
            model.free_memory()

        def try_batch():

            for t in range(times):
                xb = torch.randint(0, model.vocab_size, (batch_size, model.block_size), dtype=torch.long).to(model.device)
                yb = torch.randint(0, model.vocab_size, (batch_size, model.block_size), dtype=torch.long).to(model.device)

                logits,loss = model(xb,yb)
                del logits

                if optimizer:
                    model.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                del loss



        batch_size = starting_size
        delta_size = min(delta_size, 0.9)

        while batch_size > 0:

            try:
                cleanup()

                print(f"Trying batch_size {batch_size}...", end='')
                
                if optimizer is not None:
                    try_batch()
                else:
                    with torch.no_grad():
                        try_batch()

                print(f" Fits")
                break

            except Exception as e:
                print(f" Out of memory")
                # print(e)
    
                if delta_size > 0:
                    new_size = int(batch_size * delta_size)
                else:
                    new_size = int(batch_size + delta_size)

                new_size = max(0, new_size)

                if new_size == batch_size: # if same, decrement by one
                    batch_size -= 1
                else:
                    batch_size = new_size

        try:
            cleanup()
        except Exception as e:
            print("Final cleanup() raised exception:", e)
            pass

        print(f"Enough memory for batch_size {batch_size}")
        return batch_size
