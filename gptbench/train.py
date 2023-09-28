"""

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

        # these are not config settings but stored counts/best loss information'
        c.setup('sample_num', 0, int, 'The number of trained samples so far')

        c.setup('train_loss', float('inf'), float, 'Last evaluated train dataset loss')

        c.setup('val_loss', float('inf'), float, 'Last evaluated validation dataset loss')

        c.setup('eval_loss', float('inf'), float, 'Last evaluation loss calculated from train_loss and val_loss according to eval_type')


        c.setup('eval_period', 100, int, 'In batch iterations: each n batches we eval and check if saving model. 0 for none')

        c.setup('eval_type', 1.0, float, 'How to estimate loss -> 0: on train data, 1: on val data (or train if no val dataset), ]0,1[: weighted average of train and val (or train only if no val dataset')

        c.setup('eval_iters', 100, int, 'Count of batch_size iterations for loss evaluation')

        c.setup('eval_save_checkpt', 1, int, 'When to save a checkpoint: 0=never, 1=on new lower evaluated loss, 2=always')

        c.setup('eval_save_loss', 'csv,tensorboard', str, "Multiple values allowed: 'csv' saves a loss.csv, 'tensorboard' creates tensorboard logs")


        c.setup('sample_period', -10., float, 'When to sample, in batch iterations. 0=never. Negative means -multiples of eval_period')


        c.setup('log_period', -0.1, float, 'Simple forward pass loss log in batch iterations. Negative means -multiples of eval_period')

        return c




    def __init__(self, name=DEFAULT_NAME, work_dir=DEFAULT_WORK_DIR, log_mask=LogFlag.ALL):
        super().__init__(name, work_dir, log_mask)

        self.trainer = None
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




    def train(self, 
              trainer_batch_size=None, 
              iter_count=None, batch_end_callback=None, **over_train_config_kwargs):

        """  """

        self.log(LogFlag.INIT, f"Training")

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

        if self.config.train.log_period < 0:
            self._log_period = max(self.config.train.eval_period, 
                                   int(self.config.train.eval_period * -self.config.train.log_period))
        else:
            self._log_period = self.config.train.log_period

        if self.config.train.sample_period < 0:
            self._sample_period = max(self.config.train.eval_period, 
                                      int(self.config.train.eval_period * -self.config.train.sample_period))
        else:
            self._sample_period = self.config.train.sample_period

        self._last_saved_eval_loss = self.config.train.eval_loss



        if self.in_log(LogFlag.CUDA_MEMORY):
            cuda_max_memory_init()



        if self.trainer is None:
            # construct the trainer object
            self.trainer = Trainer(self.config.trainer, 
                                   self.train_dataset, 
                                   self.model, 
                                   start_sample_num=self.config.train.sample_num,
                                   optimizer=None, optimizer_state_dict=self._resumed_optimizer_state_dict)

            if self._resumed_optimizer_state_dict is not None:
                self.log(LogFlag.INIT , "Resuming optimizer state")
                self._resumed_optimizer_state_dict = None # consummed!


        self.log(LogFlag.INIT, f"Batches per epoch: {int(self.trainer.batches_for_epoch())}")


        if self.config.train.eval_save_loss is not None:

            if 'csv' in self.config.train.eval_save_loss:
                iter_num = Trainer.iter_from_sample(self.config.train.sample_num, 
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
        self.trainer.run(run_iter_count=iter_count)

        # restore saved config: but update new values for sample_num, and 3 losses
        saved_train_config.sample_num = self.config.train.sample_num
        saved_train_config.train_loss = self.config.train.train_loss
        saved_train_config.val_loss = self.config.train.val_loss
        saved_train_config.eval_loss = self.config.train.eval_loss

        self.config.train = saved_train_config




    # iteration callback
    @staticmethod
    def default_batch_end_callback(trainer, train):

        """
        train callback state:
            _eval_period
            _log_period
            _sample_period

            _last_saved_eval_loss

        """

        train_config = train.config.train

        train_config.sample_num = trainer.sample_num
        iter_num = trainer.get_iter_num()

        first_iter = (iter_num == trainer.get_start_iter_num())


        if train._log_period and iter_num % train._log_period == 0:
            train.log(LogFlag.TRAIN_ITER, f"iter {iter_num} loss={trainer.last_loss:.4f}, iter_dt={trainer.iter_dt * 1000:.2f}ms")

        # evaluate model? And save checkpoint, loss, etc
        if (train._eval_period and 
            (iter_num == 0 or not first_iter) and # don't eval on first_iter except if iter 0
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
            train_config.train_loss = train_loss
            train_config.val_loss = val_loss
            train_config.eval_loss = eval_loss

            train.log(LogFlag.TRAIN_EVAL, f"iter {iter_num} ({trainer.epoch_from_sample_num():.3f} epoch): loss train={train_loss:.4f}, val={val_loss:.4f}, eval->{eval_loss:.4f}")


            if (train_config.eval_save_checkpt == 1 and eval_loss < train._last_saved_eval_loss) \
               or train_config.eval_save_checkpt == 2: # save a checkpoint

                train.log(LogFlag.TRAIN_EVAL, f"==> Saving model at iter={iter_num}, eval loss->{eval_loss:.4f} ")

                # @ATTN: trainer.sample_num is already the sample num of next batch, which is okay
                train.save_checkpoint()

                train._last_saved_eval_loss = eval_loss


            if train_config.eval_save_loss is not None:
                if 'csv' in train_config.eval_save_loss:
                    loss_append(train.log_path, [(iter_num, train_loss, val_loss)] )

                if 'tensorboard' in train_config.eval_save_loss:
                    train._tensorboard_writer.add_scalar('Loss/train', train_loss, iter_num)
                    train._tensorboard_writer.add_scalar('Loss/val', val_loss, iter_num)


            if train._sample_period and iter_num % train._sample_period == 0:
                train.sample(train.config.sample.start_text)
                model_evaluated = True


        train.log(LogFlag.TRAIN_DOT, '.', end='', flush=True)

        if first_iter:
            train.log(LogFlag.CUDA_MEMORY, cuda_max_memory())






   # -----------------------------------------------------------------------------
    def save_checkpoint(self):

        from .sample import Sample

        self.ensure_path()

        checkpoint_save(self.path, 
                        self.model, self.trainer.optimizer,

                        self.config.sample.to_dict(include_non_jsonable=False),
                        self.config.train.to_dict(include_non_jsonable=False),

                        self.config.model.to_dict(include_non_jsonable=False),
                        self.config.dataset.to_dict(include_non_jsonable=False),
                        self.config.trainer.to_dict(include_non_jsonable=False)
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



