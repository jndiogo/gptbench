"""

"""

import os, sys, copy, signal, json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from gptbench.model import GPT
from gptbench.trainer import Trainer
from gptbench.dataset import GPT2TokensDataset, DatasetBase
from gptbench.utils import CfgNode, set_seed, last_config_save, die, print_sepline, cuda_max_memory_init, cuda_max_memory_print
from gptbench.sample import sample




# -----------------------------------------------------------------------------
def train_get_default_config():

    # train.*
    c = CfgNode()

    c.start_iter_num = 0
    c.start_loss = float('inf')

    c.log_period = -10 # simple forward pass loss log. Negative numbers mean max(1, int(eval_period/-log_period))

    c.eval_period = 100 # each n batches we eval and check if saving model. 0 for none
    c.eval_type = 2 # how to estimate loss -> 1: on test data, 2: on val data (or test if no val dataset), 1|2=3: mean(test,val)
    c.eval_iters = 100
    c.eval_save_checkpt = 1 # 0=never, 1=on lower loss, 2=always

    c.sample_period = 1000 # when to sample. 0 for never

    c.debug = 0 # 0: none, 1: log cuda peak used memory on each evaluation

    return c


def train_checkpoint_config_keys():
    return ["eval_period", "eval_type", "eval_iters", "sample_period", "start_iter_num", "start_loss"]






# -----------------------------------------------------------------------------
CHECKPOINT_VERSION = 1

def checkpoint_load(path_prefix, load_optimizer_state):
    """ """

    model_state_dict = torch.load(path_prefix + ".pt")
    if load_optimizer_state:
        optimizer_state_dict = torch.load(path_prefix + ".opti")
    else:
        optimizer_state_dict = None

    with open(path_prefix + '.json', 'r', encoding='utf-8') as f:
        js = f.read()
    j = json.loads(js)

    return (model_state_dict, optimizer_state_dict, 

            j['train'], 
            j['model'], 
            j['trainer'],
            j['dataset'] )



def checkpoint_save(path_prefix, 
                    model, optimizer, 

                    train_config_dict,
                    model_config_dict,
                    trainer_config_dict,
                    dataset_config_dict):

    # no CTRL+C interruptions while saving, please (malformed checkpoint files)
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


    torch.save(model.state_dict(), path_prefix + ".pt")
    torch.save(optimizer.state_dict(), path_prefix + ".opti")

    config_info = {'_version': CHECKPOINT_VERSION,
                   'train': train_config_dict,
                   'model': model_config_dict,
                   'trainer': trainer_config_dict,
                   'dataset': dataset_config_dict
                   }

    json_str = json.dumps(config_info, indent=4)

    with open(path_prefix + '.json', 'w', encoding='utf-8') as f:
        f.write(json_str)


    # restore original handler
    signal.signal(signal.SIGINT, original_sigint)



# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(train_dataset, val_dataset, model, batch_size, iters):
    """ train_dataset or val_dataset can be None to skip its eval returns train_loss,val_loss any of which can be None"""

    model.eval()

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

            x, y = x.to(model.device), y.to(model.device)

            _, loss = model(x,y)

            losses[k] = loss.item()

        out.append(losses.mean().item())

    return out



# -----------------------------------------------------------------------------
def train(config, model, trainer = None, optimizer_state_dict = None):
    """config is global config """

    if config.train.debug & 1:
        cuda_max_memory_init()


    if trainer is None:
        # construct the trainer object
        trainer = Trainer(config.trainer, model, config.dataset, start_iter_num=config.train.start_iter_num)

        print(config.trainer, trainer.get_start_iter_num())


    if optimizer_state_dict is not None:
        print("Resuming optimizer state")
        trainer.set_optimizer_state_dict(optimizer_state_dict)

    print(f"Batches per epoch: {int(trainer.batches_for_epoch())}")

    # only worth checking for training
    if config.dataset.val is None: # no validations dataset?
        config.train.eval_type &= 1 # clear val bit
    if config.train.eval_type & 3 == 0: # force at least train
        config.train.eval_type = 1

    assert (config.train.eval_type & 3) != 0, "config.train.eval_type must be set to 1, 2 or 1|2"

    train_dataset = trainer.train_dataset

    last_saved_loss = config.train.start_loss


    if config.train.log_period < 0:
        log_period = max(1, int(config.train.eval_period/-config.train.log_period))
    else:
        log_period = config.train.log_period


    # iteration callback
    def batch_end_callback(trainer):
        nonlocal last_saved_loss

        iter_num = trainer.iter_num

        if log_period and iter_num % log_period == 0:
            print(f"iter {iter_num} | loss {trainer.last_loss:.4f} | iter_dt {trainer.iter_dt * 1000:.2f}ms")

        # report, save model?
        if iter_num >= trainer.get_start_iter_num() + 1:

            model_evaluated = False

            if config.train.eval_period and iter_num % config.train.eval_period == 0: # evaluate loss 

                train_loss, val_loss = estimate_loss(
                    train_dataset,
                    config.dataset.val,
                    model,
                    trainer.config.batch_size,
                    config.train.eval_iters)

                if config.train.eval_type & 3 == 3:
                    loss = (train_loss + val_loss) / 2.
                else:
                    loss = val_loss if (config.train.eval_type & 2) and val_loss else train_loss

                val_loss = val_loss if val_loss is not None else float('inf')

                print(f"iter {iter_num} ({trainer.epoch_from_iter_num():.3f} epoch) | loss {loss:.4f} ({train_loss:.4f},{val_loss:.4f}) | iter_dt {trainer.iter_dt * 1000:.2f}ms")

                model_evaluated = True


                if (config.train.eval_save_checkpt == 1 and loss < last_saved_loss) \
                   or config.train.eval_save_checkpt == 2: # save a checkpoint

                    print(f"==> Saving model at loss={loss:.4f} iter={iter_num}")

                    train_config = copy.copy(config.train)
                    train_config.start_iter_num = iter_num
                    train_config.start_loss = loss

                    checkpoint_save(config._model_path_prefix, 
                                    model, trainer.optimizer,

                                    train_config.to_dict(False, train_checkpoint_config_keys()),
                                    config.model.to_dict(False, GPT.checkpoint_config_keys()), 
                                    config.trainer.to_dict(False, Trainer.checkpoint_config_keys()),
                                    config.dataset.to_dict(False, DatasetBase.checkpoint_config_keys())
                                    )

                    last_saved_loss = loss

                if config.train.debug & 1:
                    cuda_max_memory_print()


            if config.train.sample_period and iter_num % config.train.sample_period == 0:
                # evaluate both the train and test score
                sample(config.sample, model, train_dataset)

                model_evaluated = True

            
            if model_evaluated:
                model.train() # revert model to training mode


        print('.', end='', flush=True)





    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()


