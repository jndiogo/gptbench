"""
Modified trainer.py from minGPT.

Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

from .conf import Conf


class Trainer:

    @staticmethod
    def get_default_config():
        c = Conf()

        c.setup('batch_size', 32, int, 'Size of the batch in each forward training iteration')

        c.setup('accum_size', None, int, 'Size for batch gradient accumulation, allowing for larger batch sizes with lower memory usage. Setting batch_size must be a multiple of accum_size. For example: batch_size=32, accum_size=4 will simulate a batch of 32 by training 8 batches of 4 rows')

        # dataloader parameters
        c.setup('n_workers', 0, int, 'DataLoader workers. In Windows setting to greater than 0 causes a long delay when calling iter().')

        c.setup('max_samples', None, int, 'Absolute maximum limit on training samples. Negative -n for number of epochs')

        c.setup('grad_norm_clip', 1.0, float, 'Clip gradients to this norm')

        # optimizer parameters
        c.setup('optimizer', 'adamw', str, "Optimizer type: 'sgd' or 'adamw'")

        c.setup('learning_rate', 1e-4, float, 'Initial learning rate') 
        #5e-4? @TODO: find reasonable value. "the model we're using is so small that we can go a bit faster"

        # these are for AdamW optimizer
        c.setup('adamw_beta1', 0.9, float, 'AdamW beta1'), 
        c.setup('adamw_beta2', 0.95, float, 'AdamW beta2'),
        c.setup('adamw_weight_decay', 0.1, float, 'AdamW weight decay, only applied on matmul weights')

        return c



    def __init__(self, trainer_config, train_dataset, model, 
                 start_sample_num = 0,
                 optimizer = None, optimizer_state_dict=None):

        self.config = trainer_config

        self.model = model
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        self.set_optimizer(optimizer, optimizer_state_dict)

        self.sample_num = self.start_sample_num = start_sample_num
        self.run_sample_num = 0

        self.run_start_time = self.run_last_dur = 0
        self.iter_last_dur = 0




    def set_optimizer(self, optimizer, optimizer_state_dict=None):
        if optimizer is None:
            self.optimizer = self.model.configure_optimizers(self.config)
        else:
            self.optimizer = optimizer

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)


    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def del_callback(self, onevent: str, callback):
        if onevent in self.callbacks:
            cbs = self.callbacks
            if callback in cbs:
                cbs.remove(callback)

    def clear_callbacks(self, onevent: str):
        if onevent in self.callbacks:
            self.callbacks.pop(onevent)

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)


    # epoch-related
    @staticmethod
    def calc_epoch_from_sample_num(sample_num, train_dataset_len):
        return sample_num / train_dataset_len

    def epoch_from_sample_num(self):
        return Trainer.calc_epoch_from_sample_num(self.sample_num, len(self.train_dataset))

    def batches_for_epoch(self):
        return len(self.train_dataset) / self.config.batch_size


    # samples: 1 iter = training of batch_size samples
    def get_iter_from_sample(self, sample_num):
        return sample_num // self.config.batch_size

    @staticmethod
    def iter_from_sample(sample_num, batch_size):
        return sample_num // batch_size

    def get_run_sample_num(self):
        """ run means inside run(): 0..sample_count: 0 before any training, 1 after first batch """
        return self.run_sample_num


    # batch iterations: integer
    def get_start_iter_num(self): # 0-based: 0 before any training, 1 after first batch
        return self.get_iter_from_sample(self.start_sample_num)

    def get_iter_num(self):  # 0-based: 0 before any training, 1 after first batch
        return self.get_iter_from_sample(self.sample_num)

    def get_run_iter_num(self):
        """ run means inside run(): 0..sample_count: 0 before any training, 1 after first batch """
        return self.get_iter_from_sample(self.run_sample_num)



    def run(self, 
            run_sample_count=None, run_iter_count=None):

        assert self.optimizer is not None, "Optimizer must already be setup"


        model, config = self.model, self.config

        if self.config.max_samples is not None:
            if self.config.max_samples < 0:
                max_samples = -self.config.max_samples * len(self.train_dataset)
            else:
                max_samples = self.config.max_samples
        else:
            max_samples = None



        if config.accum_size is None:
            acc_size = config.batch_size
        else:
            assert config.batch_size % config.accum_size == 0, f"Setting trainer.batch_size {trainer_config.batch_size} must be a multiple of trainer.accum_size {trainer_config.accum_size}"
            acc_size = config.accum_size

        acc_steps = max(config.batch_size // acc_size, 1)


        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True, #@ATTN: why True?
            batch_size=config.batch_size,
            num_workers=config.n_workers
        )

        model.train()

        # slowdown here in Windows, if num_workers > 0
        data_iter = iter(train_loader)

        self.run_sample_num = 0 # the sample number of the current train() call
        self.last_loss = float('inf')

        self.run_start_time = iter_start_time = time.time()
        self.run_last_dur = self.iter_last_dur = 0
        
        while True:


            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [t.to(model.device) for t in batch]
            x, y = batch


            if acc_steps == 1: # batch is a single forward: separated for simplicity
                # forward the model
                logits, loss = model(x, y)                
                del logits
                self.last_loss = loss.item()

                # backprop and update the parameters
                loss.backward()

            else: # gradient accumulation: scale loss and acumulate gradient
                loss_acc = 0.
                for s in range(acc_steps):

                    # split batch samples
                    begin = s * acc_size
                    sx = x[begin:begin+acc_size]
                    sy = y[begin:begin+acc_size]

                    # forward the model
                    logits, loss = model(sx, sy)
                    del logits

                    loss = loss / acc_steps # scale loss to account for gradient accumulation
                    loss_acc += loss.item()

                    # backprop and update the parameters
                    loss.backward()

                self.last_loss = loss_acc

            # arriving here an entire batch_size has been trained

            # gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

            # step the optimizer
            self.optimizer.step()

            # clean up gradients, no longer needed
            model.zero_grad(set_to_none=True)

            del loss # does this help with peak memory consumption? What about speed? Also the reason for del logits above, twice

            # update accounting
            self.sample_num += self.config.batch_size
            self.run_sample_num += self.config.batch_size

            # call callbacks
            self.trigger_callbacks('on_batch_end')
            if not model.training: # callbacks may have moved to eval mode:
                model.train()             


            tnow = time.time()
            self.iter_last_dur = tnow - iter_start_time
            iter_start_time = tnow


            # termination conditions
            if run_sample_count is not None and self.run_sample_count >= run_sample_count:
                break
            if run_iter_count is not None and self.get_run_iter_num() >= run_iter_count:
                break
            if max_samples is not None and self.sample_num >= max_samples:
                break


        self.run_last_dur = time.time() - self.run_start_time 

