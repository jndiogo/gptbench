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
            j['model'], 
            j['trainer'],
            j['dataset'],
            j['eval'], j['eval_iters'],
             j['eval_period'], j['eval_sample_period'],
            j['_iter_num'], j['_loss'] )



def checkpoint_save(path_prefix, 
                    model, optimizer, 
                    model_config_dict,
                    trainer_config_dict,
                    dataset_config_dict,
                    eval, eval_iters, eval_period, eval_sample_period,
                    _iter_num, _loss):

    # no CTRL+C interruptions while saving, please (malformed checkpoint files)
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


    torch.save(model.state_dict(), path_prefix + ".pt")
    torch.save(optimizer.state_dict(), path_prefix + ".opti")

    config_info = {'_version': CHECKPOINT_VERSION,
                   '_iter_num': _iter_num, '_loss': _loss,
                   'model': model_config_dict,
                   'trainer': trainer_config_dict,
                   'dataset': dataset_config_dict,
                   'eval': eval, 'eval_iters': eval_iters, 
                    'eval_period': eval_period, 'eval_sample_period': eval_sample_period,
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
def train(config, trainer, start_loss=float('inf')):
    """config is global config """

    assert (config.eval & 3) != 0, "config.eval must be set to 1, 2 or 1|2"

    model = trainer.model
    train_dataset = trainer.train_dataset

    last_saved_loss = start_loss

    # iteration callback
    def batch_end_callback(trainer):
        nonlocal last_saved_loss

        iter_num = trainer.iter_num

        # report, save model?
        if iter_num > trainer.get_start_iter_num():

            model_evaluated = False

            if iter_num % config.eval_period == 0: # evaluate loss 

                train_loss, val_loss = estimate_loss(
                    train_dataset,
                    config.dataset.val,
                    model,
                    trainer.config.batch_size,
                    config.eval_iters)

                if config.eval & 3 == 3:
                    loss = (train_loss + val_loss) / 2.
                else:
                    loss = val_loss if (config.eval & 2) and val_loss else train_loss

                val_loss = val_loss if val_loss is not None else float('inf')

                print(f"iter {iter_num} ({trainer.epoch_from_iter_num():.3f} epoch) | loss {loss:.4f} ({train_loss:.4f},{val_loss:.4f}) | iter_dt {trainer.iter_dt * 1000:.2f}ms")

                model_evaluated = True


                if loss < last_saved_loss: # save a checkpoint

                    print(f"==> Saving model at loss={loss:.4f} iter={iter_num}")

                    checkpoint_save(config._model_path_prefix, 
                                    model, trainer.optimizer,
                                    
                                    config.model.to_dict(False, GPT.checkpoint_config_keys()), 
                                    config.trainer.to_dict(False, Trainer.checkpoint_config_keys()),
                                    config.dataset.to_dict(False, DatasetBase.checkpoint_config_keys()),
                                    
                                    config.eval, config.eval_iters, config.eval_period, config.eval_sample_period,
                                    iter_num, loss)

                    last_saved_loss = loss


            if iter_num % config.eval_sample_period == 0:
                # evaluate both the train and test score
                sample(config.sampler, model, train_dataset)

                model_evaluated = True

            
            if model_evaluated:
                model.train() # revert model to training mode


        print('.', end='', flush=True)


        cuda_max_memory_print()



    trainer.set_callback('on_batch_end', batch_end_callback)

    cuda_max_memory_init()

    # run the optimization
    trainer.run()







