"""
"""

import sys

from gptbench.dataset import GPT2TokensDataset
import gptbench.common as common

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    c = common.default_config()

    c.eval = 2 # 1: on test data, 2: on val data (or test if none), 1|2=3: on mean(test,val)
    c.eval_iters = 100

    c.eval_period = 100 # each n batches we eval and check if saving model
    c.eval_sample_period = 300

    # dataset
    c.dataset.cls = GPT2TokensDataset
    #c.dataset.path = '../../../datasets/text/en/wikitext-2-raw/wiki.all.gpt2-tokens'

    # model

    # trainer

    # sampler
    c.sampler.eotstop = 1
    c.sampler.multiline = 0


    args_config = common.config_from_sysargv(sys.argv)
    
    common.run(c, args_config)
