"""

"""

import os, sys, copy, signal, json

import torch


from .sample import Sample, LogFlag

from .model import GPT
from .trainer import Trainer

from .config import checkpoint_save, sample_checkpoint_config_keys, train_checkpoint_config_keys, dataset_checkpoint_config_keys

from .utils import print_sepline, cuda_max_memory_init, cuda_max_memory



# -----------------------------------------------------------------------------

class Train(Sample):
    def __init__(self, name='model', work_dir='./out', log_mask=LogFlag.ALL):
        super().__init__(name, work_dir, log_mask)

        self.trainer = None
        self._can_train = True





    def train(self,
              batch_size=None, over_trainer_config=None, 
              over_train_config=None,
              **kwargs):

        """ kwargs: key value of config.train settings """


        # trainer config
        if batch_size is not None:
            self.config.trainer.batch_size = batch_size
        if over_trainer_config is not None:
            self.config.trainer.merge_from_config(over_trainer_config)

        # train config
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


        self._log(LogFlag.CUDA_MEMORY, cuda_max_memory_init())


        if self.trainer is None:
            # construct the trainer object
            self.trainer = Trainer(self.config.trainer, self.train_dataset, self.model, start_iter_num=train_config.start_iter_num)

        if self._resumed_optimizer_state_dict is not None:
            self._log(LogFlag.INIT , "Resuming optimizer state")
            self.trainer.set_optimizer_state_dict(self._resumed_optimizer_state_dict)

            self._resumed_optimizer_state_dict = None # consummed!


        self._log(LogFlag.INIT, f"Batches per epoch: {int(self.trainer.batches_for_epoch())}")

        last_saved_loss = train_config.start_eval_loss



        # iteration callback
        def batch_end_callback(trainer):
            nonlocal train_config, last_saved_loss

            iter_num = self.trainer.iter_num

            if train_config.log_period and iter_num % train_config.log_period == 0:
                self._log(LogFlag.BATCH_LOSS, f"iter {iter_num} | loss {self.trainer.last_loss:.4f} | iter_dt {self.trainer.iter_dt * 1000:.2f}ms")

            # report, save model?
            if iter_num >= self.trainer.get_start_iter_num() + 1:

                if train_config.eval_period and iter_num % train_config.eval_period == 0: # evaluate loss 

                    # evaluate both the train and validation score
                    train_loss, val_loss = self.estimate_loss(
                        self.train_dataset,
                        self.val_dataset,
                        self.config.trainer.batch_size,
                        train_config.eval_iters)

                    if train_config.eval_type & 3 == 3:
                        loss = (train_loss + val_loss) / 2.
                    else:
                        loss = val_loss if (train_config.eval_type & 2) and val_loss else train_loss

                    val_loss = val_loss if val_loss is not None else float('inf')

                    self._log(LogFlag.EVAL_LOG, f"iter {iter_num} ({self.trainer.epoch_from_iter_num():.3f} epoch) | eval loss {loss:.4f} ({train_loss:.4f}, {val_loss:.4f})")


                    if (train_config.eval_save_checkpt == 1 and loss < last_saved_loss) \
                       or train_config.eval_save_checkpt == 2: # save a checkpoint

                        self._log(LogFlag.EVAL_LOG, f"==> Saving model at loss={loss:.4f} iter={iter_num}")

                        self.save(iter_num, loss, train_loss, val_loss)

                        last_saved_loss = loss


                    self._log(LogFlag.CUDA_MEMORY, cuda_max_memory())


                if train_config.sample_period and iter_num % train_config.sample_period == 0:
                    self.sample()
                    model_evaluated = True


            self._log(LogFlag.BATCH_LOSS, '.', end='', flush=True)



        self.trainer.set_callback('on_batch_end', batch_end_callback)

        # run the optimization
        self.trainer.run()


        # update config
        train_config.start_iter_num = iter_num
        train_config.start_eval_loss = loss
        train_config.start_train_loss = train_loss
        train_config.start_val_loss = val_loss




    def save(self, start_iter_num, 
             start_eval_loss, start_train_loss, stat_val_loss):

        self._ensure_work_dir()

        dup_train_config = copy.copy(self.config.train)
        dup_train_config.start_iter_num = start_iter_num
        dup_train_config.start_eval_loss = start_loss
        dup_train_config.start_train_loss = start_train_loss
        dup_train_config.start_val_loss = start_val_loss

        checkpoint_save(self._model_path_prefix, 
                        self.model, self.trainer.optimizer,

                        dup_train_config.to_dict(False, train_checkpoint_config_keys()),
                        self.config.sample.to_dict(False, sample_checkpoint_config_keys()),

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



