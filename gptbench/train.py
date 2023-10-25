"""
The Train class can do model training and also inference as it derives from the Sample class.
"""

import os, sys, signal, json, copy

import torch
from torch.utils.tensorboard import SummaryWriter

from .sample import Sample, LogFlag, DEFAULT_NAME, DEFAULT_WORK_DIR

from .model import GPT
from .trainer import Trainer

from .config import checkpoint_save, loss_append, loss_trim

from .conf import Conf
from .utils import print_sepline, cuda_max_memory_init, cuda_max_memory



# -----------------------------------------------------------------------------

class Train(Sample):

    @staticmethod
    def get_default_config():

        # train.*
        c = Conf()

        c.setup('eval_period', 100, int, 'In batch iterations: each n batches we eval and check if saving model. 0 for none')

        c.setup('eval_type', 1.0, float, 'How to estimate loss -> 0: on train data, 1: on val data (or train if no val dataset), ]0,1[: weighted average of train and val (or train only if no val dataset')

        c.setup('eval_iters', 100, int, 'Count of batch_size iterations for loss evaluation')

        c.setup('eval_save_checkpt', 1, int, 'When to save a checkpoint: 0=never, 1=on new lower evaluated loss, 2=always')

        c.setup('eval_save_loss', 'csv,tensorboard', str, "Multiple values allowed, separated with comma: 'csv' saves a loss.csv, 'tensorboard' creates tensorboard logs")


        return c




    def __init__(self, name=DEFAULT_NAME, work_dir=DEFAULT_WORK_DIR, 
                 seed=None,
                 log_mask=LogFlag.ALL, 
                 log_dot_period=-0.01, log_loss_period=-0.1, log_sample_period=-10.0):

        super().__init__(name, work_dir, seed, log_mask)

        self.set_train_log_periods(log_dot_period, log_loss_period, log_sample_period)

        self._can_train = True
        self._tensorboard_writer = None



    def update_train_config(self, over_train_config=None, **over_train_config_kwargs):
        if over_train_config is not None:
            self.config.train.update(over_train_config)
        # override existing keys from kwargs
        self.config.train.update(over_train_config_kwargs)


    def update_trainer_config(self, over_trainer_config=None, **over_trainer_config_kwargs):
        if over_trainer_config is not None:
            self.config.trainer.update(over_trainer_config)
        # override existing keys from kwargs
        self.config.trainer.update(over_trainer_config_kwargs)


    def set_train_log_periods(self, dot_period=None, loss_period=None, sample_period=None):
        """
        A positive value is an iteration count, negative value is -ratio of config.train.eval_period
        loss_period: simple forward pass loss log in batch iterations. Negative means -multiples of eval_period
        dot_period: log a . for each n batch iterations. Negative means -multiples of eval_period
        sample_period: when to sample, in batch iterations. 0=never. Negative means -multiples of eval_period
        """
        if dot_period is not None:
            self.log_dot_period = dot_period

        if loss_period is not None:
            self.log_loss_period = loss_period 

        if sample_period is not None:
            self.log_sample_period = sample_period


    def train(self, 
              trainer_batch_size=None, 
              iter_count=None, batch_end_callback=None, **over_train_config_kwargs):

        """ Dataset must have been set either by calling set_datasets() or via config """


        # save train config so that any overrides are local to this function
        saved_train_config = copy.copy(self.config.train)


        # trainer config ----------------------------------------------------
        if trainer_batch_size is not None:
            self.config.trainer.batch_size = trainer_batch_size

        # train config ------------------------------------------------------
        #override existing config.train keys from kwargs
        self.config.train.update(over_train_config_kwargs)


        # sanity checks
        assert self.config.train.eval_type >= 0. and self.config.train.eval_type <= 1., "config.train.eval_type must be >= 0.0 and <= 1.0"


        # prepare state for callback
        self._eval_period = self.config.train.eval_period        

        if self.log_sample_period < 0:
            self.log_sample_period = max(1, int(self._eval_period * -self.log_sample_period))

        if self.log_loss_period < 0:
            self.log_loss_period = max(1, int(self._eval_period * -self.log_loss_period))

        if self.log_dot_period < 0:
            self.log_dot_period = max(1, int(self._eval_period * -self.log_dot_period))


        # ensure model and logs dirs exist
        if self.config.train.eval_save_loss:
            self.ensure_path()            


        if self.in_log(LogFlag.CUDA_MEMORY):
            cuda_max_memory_init()



        if self.train_dataset is None: # load dataset(s) from config
            (self.train_dataset, self.val_dataset) = self._load_datasets()
            assert self.train_dataset is not None, "Unable to load dataset(s)"


        if self.trainer is None:
            # construct the trainer object
            self.trainer = Trainer(self.config.trainer, 
                                   self.train_dataset, 
                                   self.model, 
                                   start_sample_num=self.state['n_samples'],
                                   optimizer=None, optimizer_state_dict=self._loaded_optimizer_state_dict)

            if self._loaded_optimizer_state_dict is not None:
                self.log(LogFlag.INIT , "Resumed optimizer state")
                self._loaded_optimizer_state_dict = None # consummed!



        if self.config.train.eval_save_loss is not None:

            if 'csv' in self.config.train.eval_save_loss:
                iter_num = Trainer.iter_from_sample(self.state['n_samples'], 
                                                    self.config.trainer.batch_size)
                # trim loss at iter_num
                loss_trim(self.log_path, iter_num if iter_num > 0 else None)

            if 'tensorboard' in self.config.train.eval_save_loss:
                if self._tensorboard_writer:
                    self._tensorboard_writer.close()
                self._tensorboard_writer = SummaryWriter(log_dir=self.log_path)


        if batch_end_callback is None:
            batch_end_callback = Train.default_batch_end_callback

        self.trainer.set_callback('on_batch_end', lambda trainer: batch_end_callback(trainer, self))


        # run the optimization
        self.log(LogFlag.INIT, f"Training") # ({int(self.trainer.batches_for_epoch())} batch iters/epoch)

        self.trainer.run(run_iter_count=iter_count)


        self.config.train = saved_train_config




    # iteration callback
    @staticmethod
    def default_batch_end_callback(trainer, train):

        """
        """

        train_config = train.config.train

        train.state['n_samples'] = trainer.sample_num
        iter_num = trainer.get_iter_num()

        first_iter = (iter_num == trainer.get_start_iter_num())


        # evaluate model? And save checkpoint, loss, etc
        if (train._eval_period and 
            (iter_num == 0 or not first_iter) and # don't eval on local first_iter except if iter 0
            iter_num % train._eval_period == 0): # evaluate train/val loss 

            # evaluate both the train and validation score
            train_loss, val_loss = train.estimate_loss(
                train.train_dataset,
                train.val_dataset,
                train.config.trainer.batch_size,
                train_config.eval_iters)

            if val_loss is None: # no validation dataset present
                eval_loss = train_loss
                val_loss = float('inf')
            else:
                eval_loss = train_loss * (1. - train_config.eval_type) + val_loss * train_config.eval_type            

            # update config after evaluation
            train.state['train_loss'] = train_loss
            train.state['val_loss'] = val_loss
            train.state['eval_loss'] = eval_loss

            if train.log_dot_period and not first_iter:
                train.log(LogFlag.TRAIN_ITER, '') # new line after ......

            train.log(LogFlag.TRAIN_EVAL, f"Iter {iter_num} ({trainer.epoch_from_sample_num():.3f} epoch): loss train={train_loss:.4f}, val={val_loss:.4f}, eval->{eval_loss:.4f}")


            if (train_config.eval_save_checkpt == 1 and eval_loss < train.last_saved_state['eval_loss']) \
               or train_config.eval_save_checkpt == 2: # save a checkpoint

                train.log(LogFlag.TRAIN_EVAL, f"==> Saving model at iter={iter_num}, eval loss->{eval_loss:.4f} ")

                # @ATTN: trainer.sample_num is already the sample num of next batch, which is okay
                train.save()


            if train_config.eval_save_loss is not None:
                if 'csv' in train_config.eval_save_loss:
                    loss_append(train.log_path, [(iter_num, train_loss, val_loss)] )

                if 'tensorboard' in train_config.eval_save_loss:
                    train._tensorboard_writer.add_scalar('Loss/train', train_loss, iter_num)
                    train._tensorboard_writer.add_scalar('Loss/val', val_loss, iter_num)

        else: # these only log if no eval occurred

            if train.log_loss_period and iter_num % train.log_loss_period == 0:
                if train.log_dot_period:
                    train.log(LogFlag.TRAIN_ITER, '') # new line after ......
                train.log(LogFlag.TRAIN_ITER, f"Iter {iter_num} loss={trainer.last_loss:.4f}, iter_dt={trainer.iter_dt * 1000:.2f}ms")


        if train.log_dot_period and iter_num % train.log_dot_period == 0:
            train.log(LogFlag.TRAIN_ITER, '.', end='', flush=True)


        if train.log_sample_period and iter_num % train.log_sample_period == 0:
            out=[]
            train.sample(train.config.sample.start_text, dest=out)
            train.log(LogFlag.TRAIN_ITER, 'Sampling:', '\n'.join(out))


        if first_iter:
            train.log(LogFlag.CUDA_MEMORY, cuda_max_memory())





   # -----------------------------------------------------------------------------
    def save(self, name=None):

        if self.trainer is None:
            if self._loaded_optimizer_state_dict is not None: # loaded
                optimizer_state_dict = self._loaded_optimizer_state_dict
            else:
                raise RuntimeError("Must load() or train() before saving")
        else:
            optimizer_state_dict = self.trainer.optimizer.state_dict()

        if name is not None:
            self.set_name(name)

        self.ensure_path()

        config = self.config.to_dict()
        # don't include seed
        del config['seed']

        checkpoint_save(self.path, 
                        self.state, config,
                        self.model.state_dict(), 
                        optimizer_state_dict)

        self.last_saved_state = copy.copy(self.state)


