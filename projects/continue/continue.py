"""
"""

import sys

from gptbench import empty_config, config_run

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    c = empty_config()

    # train

    # sample
    c.sample.eotstop = 1

    # dataset
    c.dataset.class_name = 'gpt2'

    # model

    # trainer


    obj = config_run(c, sys.argv)