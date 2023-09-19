"""

"""

import os, sys, copy, signal, json

import torch


from .sample import Sample, LogFlag

from .model import GPT
from .trainer import Trainer

from .config import checkpoint_save, dataset_checkpoint_config_keys

from .utils import CfgNode, print_sepline, cuda_max_memory_init, cuda_max_memory



# -----------------------------------------------------------------------------

class Train(Sample):

    @staticmethod
    def get_default_config():

        # train.*
        c = CfgNode()

        c.start_iter_num = 0
        c.start_eval_loss = float('inf')
        c.start_train_loss = float('inf')
        c.start_val_loss = float('inf')

        c.log_period = -0.1 # simple forward pass loss log. Negative numbers mean max(1, int(eval_period * -log_period))

        c.eval_period = 100 # each n batches we eval and check if saving model. 0 for none
        c.eval_type = 2 # how to estimate loss -> 1: on test data, 2: on val data (or test if no val dataset), 1|2=3: mean(test,val)
        c.eval_iters = 100
        c.eval_save_checkpt = 1 # 0=never, 1=on lower loss, 2=always

        c.sample_period = 1000 # when to sample. 0 for never

        c.batch_end_callback = 'default' # 'default', None, or callback - see Train.default_batch_end_callback()

        return c

    @staticmethod
    def checkpoint_config_keys():
        return ['start_iter_num', 'start_eval_loss', 'start_train_loss', 'start_val_loss', 'eval_period', 'eval_type', 'eval_iters', 'sample_period']




    def __init__(self, name='model', work_dir='./out', log_mask=LogFlag.ALL):
        super().__init__(name, work_dir, log_mask)

        self.trainer = None
        self._can_train = True



    def train(self,
              batch_size=None, iter_num=None, over_trainer_config=None, 
              batch_end_callback=None, over_train_config=None, **kwargs):

        """ kwargs: key value of config.train settings """


        # trainer config ----------------------------------------------------
        if batch_size is not None:
            self.config.trainer.batch_size = batch_size
        if iter_num is not None:
            self.config.trainer.iter_num = iter_num
        if over_trainer_config is not None:
            self.config.trainer.merge_from_config(over_trainer_config)

        # train config ------------------------------------------------------
        if batch_end_callback is not None:
            self.config.train.batch_end_callback = batch_end_callback
        if over_train_config is not None:
            self.config.train.merge_from_config(over_train_config)
        #override existing keys from kwargs
        self.config.train.merge_from_dict(kwargs, existing_only=True)


        # resolve train config
        if self.val_dataset is None: # no validations dataset?
            self.config.train.eval_type &= 1 # clear val bit
        if self.config.train.eval_type & 3 == 0: # force at least train
            self.config.train.eval_type = 1

        assert (self.config.train.eval_type & 3) != 0, "config.train.eval_type must be set to 1, 2 or 1|2"

        if self.config.train.log_period < 0:
            self.config.train.log_period = max(1, int(self.config.train.eval_period * -self.config.train.log_period))
        else:
            self.config.train.log_period = self.config.train.log_period


        train_config = self.config.train


        if self.in_log(LogFlag.CUDA_MEMORY):
            cuda_max_memory_init()


        if self.trainer is None:
            # construct the trainer object
            self.trainer = Trainer(self.config.trainer, 
                                   self.train_dataset, 
                                   self.model, 
                                   start_iter_num=train_config.start_iter_num,
                                   optimizer=None, optimizer_state_dict=self._resumed_optimizer_state_dict)

            if self._resumed_optimizer_state_dict is not None:
                self.log(LogFlag.INIT , "Resuming optimizer state")
                self._resumed_optimizer_state_dict = None # consummed!


        self.log(LogFlag.INIT, f"Batches per epoch: {int(self.trainer.batches_for_epoch())}")



        if self.config.train.batch_end_callback is not None:
            batch_end_callback = self.config.train.batch_end_callback

            if batch_end_callback == 'default':
                batch_end_callback = Train.default_batch_end_callback

            state = {'train': self, 'last_saved_loss': train_config.start_eval_loss}

            self.trainer.add_callback('on_batch_end', lambda trainer: batch_end_callback(trainer, state))


        # run the optimization
        self.trainer.run()


        # update config
        train_config.start_iter_num = self.trainer.iter_num
        train_config.start_eval_loss = loss
        train_config.start_train_loss = train_loss
        train_config.start_val_loss = val_loss




    # iteration callback
    @staticmethod
    def default_batch_end_callback(trainer, state):

        """
        state = {'train': Train object, 
                 'last_saved_loss': train_config.start_eval_loss} 
        """

        train_self = state['train']
        train_config = train_self.config.train


        iter_num = trainer.iter_num

        if train_config.log_period and iter_num % train_config.log_period == 0:
            train_self.log(LogFlag.BATCH_LOSS, f"iter {iter_num} | loss {trainer.last_loss:.4f} | iter_dt {trainer.iter_dt * 1000:.2f}ms")

        # report, save model?
        if iter_num >= trainer.get_start_iter_num() + 1:

            if train_config.eval_period and iter_num % train_config.eval_period == 0: # evaluate loss 

                # evaluate both the train and validation score
                train_loss, val_loss = train_self.estimate_loss(
                    train_self.train_dataset,
                    train_self.val_dataset,
                    train_self.config.trainer.batch_size,
                    train_config.eval_iters)

                if train_config.eval_type & 3 == 3:
                    loss = (train_loss + val_loss) / 2.
                else:
                    loss = val_loss if (train_config.eval_type & 2) and val_loss else train_loss

                val_loss = val_loss if val_loss is not None else float('inf')

                train_self.log(LogFlag.EVAL_LOG, f"iter {iter_num} ({trainer.epoch_from_iter_num():.3f} epoch) | eval loss {loss:.4f} ({train_loss:.4f}, {val_loss:.4f})")


                if (train_config.eval_save_checkpt == 1 and loss < state['last_saved_loss']) \
                   or train_config.eval_save_checkpt == 2: # save a checkpoint

                    train_self.log(LogFlag.EVAL_LOG, f"==> Saving model at loss={loss:.4f} iter={iter_num}")

                    train_self.save(iter_num, loss, train_loss, val_loss)

                    state['last_saved_loss'] = loss


                train_self.log(LogFlag.CUDA_MEMORY, cuda_max_memory())


            if train_config.sample_period and iter_num % train_config.sample_period == 0:
                train_self.sample()
                model_evaluated = True


        train_self.log(LogFlag.BATCH_LOSS, '.', end='', flush=True)





   # -----------------------------------------------------------------------------
    def save(self, start_iter_num, 
             start_eval_loss, start_train_loss, start_val_loss):

        from .sample import Sample

        self._ensure_work_dir()

        dup_train_config = copy.copy(self.config.train)
        dup_train_config.start_iter_num = start_iter_num
        dup_train_config.start_eval_loss = start_eval_loss
        dup_train_config.start_train_loss = start_train_loss
        dup_train_config.start_val_loss = start_val_loss

        checkpoint_save(self._model_path_prefix, 
                        self.model, self.trainer.optimizer,

                        self.config.sample.to_dict(False, Sample.checkpoint_config_keys()),
                        dup_train_config.to_dict(False, Train.checkpoint_config_keys()),

                        self.config.model.to_dict(False, GPT.checkpoint_config_keys()), 
                        self.config.dataset.to_dict(False, dataset_checkpoint_config_keys()),
                        self.config.trainer.to_dict(False, Trainer.checkpoint_config_keys())
                        )







   # -----------------------------------------------------------------------------
    @torch.no_grad()
    def estimate_loss(self, train_dataset, val_dataset, batch_size, iters):
        """ train_dataset or val_dataset can be None to skip its eval returns train_loss,val_loss any of which can be None"""

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



