"""
"""

import sys

from gptbench.dataset import GPT2TokensDataset
import gptbench.do as do

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    c = do.empty_config()

    # sample
    c.sample.eotstop = 1
    c.sample.multiline = 0

    # dataset
    c.dataset.class_name = 'gpt2'

    # model

    # trainer



    c = do.merge_config_from_sysargv(sys.argv, c)
    do.run(c)
