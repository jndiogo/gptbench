"""
Trains a character-level language model.
"""

import os, sys, copy

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.dataset import CharDataset
from mingpt.utils import set_seed, setup_work_dir, checkpoint_load, checkpoint_save, config_save, CfgNode, OneDashArgs

import common

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    c = CfgNode()

    c.seed = 0 # 0 means random seed
    c.work_dir = './out/chargpt'
    c.train_period = 100 # each n batches we report and check if saving model
    c.sample_period = 5000

    # data
    c.data = CharDataset.get_default_config()
    c.data.dataset_path = '../../../../datasets/text/en/wikitext-2-raw/wiki.all.raw'
    c.data.block_size = 128

    # model
    c.model = GPT.get_default_config()
    c.model.n_layer = 8
    c.model.n_head = 8
    c.model.n_embd = 256 # multiple of n_head

    # trainer
    c.trainer = Trainer.get_default_config()
    c.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster


    common.run_gpt(c, sys.argv[1:])

