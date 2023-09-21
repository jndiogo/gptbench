"""

"""

import os, sys, copy, signal, json

import torch


from .sample import Sample, LogFlag, DEFAULT_NAME, DEFAULT_WORK_DIR

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

        c.sample_num = 0 # total trained samples

        c.eval_loss = float('inf') # last evaluation loss according to eval_type
        c.train_loss = float('inf') # last evaluated train dataset loss
        c.val_loss = float('inf') # last evaluated validation dataset loss

        c.log_period = -0.1 # in batch iters: simple forward pass loss log. Negative numbers mean max(1, int(eval_period * -log_period))

        c.eval_period = 100 # in batch iters: each n batches we eval and check if saving model. 0 for none
        c.eval_type = 2 # how to estimate loss -> 1: on test data, 2: on val data (or test if no val dataset), 1|2=3: mean(test,val)
        c.eval_iters = 100
        c.eval_save_checkpt = 1 # 0=never, 1=on lower loss, 2=always

        c.sample_period = 1000 # in batch_iters: when to sample. 0 for never

        c.batch_end_callback = 'default' # 'default', None, or callback - see Train.default_batch_end_callback()

        return c


    @staticmethod
    def checkpoint_config_keys():
        return ['sample_num', 'eval_loss', 'train_loss', 'val_loss', 'log_period', 'eval_period', 'eval_type', 'eval_iters', 'eval_save_checkpt', 'sample_period']




    def __init__(self, name=DEFAULT_NAME, work_dir=DEFAULT_WORK_DIR, log_mask=LogFlag.ALL):
        super().__init__(name, work_dir, log_mask)

        self.trainer = None
        self._can_train = True



    def set_train_config(self, over_train_config=None, **over_train_config_kwargs):
        if over_train_config is not None:
            self.config.train.merge_from_config(over_train_config)
        # override existing keys from kwargs
        self.config.train.merge_from_dict(over_train_config_kwargs, existing_only=True)


    def set_trainer_config(self, over_trainer_config=None, **over_trainer_config_kwargs):
        if over_trainer_config is not None:
            self.config.trainer.merge_from_config(over_trainer_config)
        # override existing keys from kwargs
        self.config.trainer.merge_from_dict(over_trainer_config_kwargs, existing_only=True)




    def train(self, 
              trainer_batch_size=None, 
              iter_count=None, batch_end_callback=None, **over_train_config_kwargs):

        """  """


        # trainer config ----------------------------------------------------
        if trainer_batch_size is not None:
            self.config.trainer.batch_size = trainer_batch_size

        # train config ------------------------------------------------------
        if batch_end_callback is not None:
            self.config.train.batch_end_callback = batch_end_callback
        #override existing config.train keys from kwargs
        self.config.train.merge_from_dict(over_train_config_kwargs, existing_only=True)



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
                                   start_sample_num=train_config.sample_num,
                                   optimizer=None, optimizer_state_dict=self._resumed_optimizer_state_dict)

            if self._resumed_optimizer_state_dict is not None:
                self.log(LogFlag.INIT , "Resuming optimizer state")
                self._resumed_optimizer_state_dict = None # consummed!


        self.log(LogFlag.INIT, f"Batches per epoch: {int(self.trainer.batches_for_epoch())}")



        if self.config.train.batch_end_callback is not None:
            batch_end_callback = self.config.train.batch_end_callback

            if batch_end_callback == 'default':
                batch_end_callback = Train.default_batch_end_callback

            state = {'train': self, 'last_saved_loss': train_config.eval_loss}

            self.trainer.set_callback('on_batch_end', lambda trainer: batch_end_callback(trainer, state))


        # run the optimization
        self.trainer.run(run_iter_count=iter_count)





    # iteration callback
    @staticmethod
    def default_batch_end_callback(trainer, state):

        """
        state = {'train': Train object, 
                 'last_saved_loss': train_config.eval_loss} 
        """

        train_self = state['train']
        train_config = train_self.config.train

        if 'first' not in state:
            state['first'] = True

        train_config.sample_num = trainer.sample_num
        iter_num = trainer.get_iter_num()

        if train_config.log_period and iter_num % train_config.log_period == 0:
            train_self.log(LogFlag.BATCH_LOSS, f"iter {iter_num} | loss {trainer.last_loss:.4f} | iter_dt {trainer.iter_dt * 1000:.2f}ms")

        # report, save model?
        if iter_num >= trainer.get_start_iter_num() + 1:

            if train_config.eval_period and iter_num % train_config.eval_period == 0: # evaluate train/val loss 

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

                # update config after evaluation
                train_config.eval_loss = loss
                train_config.train_loss = train_loss
                train_config.val_loss = val_loss

                train_self.log(LogFlag.EVAL_LOG, f"iter {iter_num} ({trainer.epoch_from_sample_num():.3f} epoch) | eval loss {loss:.4f} (train={train_loss:.4f}, val={val_loss:.4f})")


                if (train_config.eval_save_checkpt == 1 and loss < state['last_saved_loss']) \
                   or train_config.eval_save_checkpt == 2: # save a checkpoint

                    train_self.log(LogFlag.EVAL_LOG, f"==> Saving model at loss={loss:.4f} iter={iter_num}")

                    # @ATTN: trainer.sample_num is already the sample num of next batch, which is okay
                    train_self.save_checkpoint()

                    state['last_saved_loss'] = loss


                if state['first']:
                    state['first'] = False
                    train_self.log(LogFlag.CUDA_MEMORY, cuda_max_memory())


            if train_config.sample_period and iter_num % train_config.sample_period == 0:
                train_self.sample()
                model_evaluated = True


        train_self.log(LogFlag.BATCH_LOSS, '.', end='', flush=True)





   # -----------------------------------------------------------------------------
    def save_checkpoint(self):

        from .sample import Sample

        self._ensure_work_dir()

        checkpoint_save(self.path_prefix, 
                        self.model, self.trainer.optimizer,

                        self.config.sample.to_dict(False, Sample.checkpoint_config_keys()),
                        self.config.train.to_dict(False, Train.checkpoint_config_keys()),

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



