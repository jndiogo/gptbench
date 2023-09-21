"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

from .utils import CfgNode

class Trainer:

    @staticmethod
    def get_default_config():
        c = CfgNode()

        # dataloader parameters
        c.n_workers = 4

        c.batch_size = 32

        c.max_samples = None # absolute maximum samples. -n for n of epochs 

        c.grad_norm_clip = 1.0

        # optimizer parameters
        c.opti = 1 # 0: SGD, 1: AdamW

        c.learning_rate = 1e-4 #5e-4? @TODO: find reasonable value. "the model we're using is so small that we can go a bit faster"

        # these are for AdamW optimizer
        c.adamw_betas = (0.9, 0.95)
        c.adamw_weight_decay = 0.1 # only applied on matmul weights


        return c

    @staticmethod
    def checkpoint_config_keys():
        return ['batch_size', 'max_samples', 'opti', 'learning_rate', 'adamw_betas', 'adamw_weight_decay', 'grad_norm_clip']


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

        self.iter_time = 0.0
        self.iter_dt = 0.0


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


    def epoch_from_sample_num(self):
        return self.sample_num / len(self.train_dataset)

    def batches_for_epoch(self):
        return len(self.train_dataset) / self.config.batch_size


    # samples: 1 iter = batch_size samples
    def get_local_sample_num(self):
        """ local means since start_sample_num """
        return self.sample_num - self.start_sample_num

    def get_run_sample_num(self):
        """ run means inside run(): 0..sample_count """
        return self.run_sample_num

    def get_iter_from_sample(self, sample_num):
        return sample_num // self.config.batch_size

    @staticmethod
    def iter_from_sample(sample_num, batch_size):
        return sample_num // batch_size


    # batch iterations: always integer
    def get_start_iter_num(self): # 0-based
        return self.get_iter_from_sample(self.start_sample_num)

    def get_iter_num(self): # 0-based
        return self.get_iter_from_sample(self.sample_num)

    def get_run_iter_num(self):
        """ run means inside run(): 0..iter_count """
        return self.get_iter_from_sample(self.run_sample_num)



    def run(self, 
            run_sample_count=None, run_iter_count=None):

        assert self.optimizer is not None, "Optimizer must already be setup"

        model, config = self.model, self.config

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.n_workers,
        )

        model.train()

        data_iter = iter(train_loader)


        if self.config.max_samples is not None:
            if self.config.max_samples < 0:
                max_samples = -self.config.max_samples * len(self.train_dataset)
            else:
                max_samples = self.config.max_samples
        else:
            max_samples = None


        self.run_sample_num = 0
        self.last_loss = float('inf')
        self.iter_time = time.time()
        
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [t.to(model.device) for t in batch]
            x, y = batch

            # forward the model
            logits, loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

            self.optimizer.step()

            self.last_loss = loss.item()


            self.trigger_callbacks('on_batch_end')

            if not model.training: # callbacks may have moved to eval mode:
                model.train()             

            self.sample_num += self.config.batch_size
            self.run_sample_num += self.config.batch_size

            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if run_sample_count is not None and self.run_sample_count >= run_sample_count:
                break
            if run_iter_count is not None and self.get_run_iter_num() >= run_iter_count:
                break
            if max_samples is not None and self.sample_num >= max_samples:
                break

