"""
"""

import sys

from gptbench.dataset import GPT2TokensDataset
import gptbench.common as common

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    c = common.empty_config()

    # dataset
    c.dataset.cls = GPT2TokensDataset

    # model

    # trainer

    # sampler
    c.sampler.eotstop = 1
    c.sampler.multiline = 0



    c = common.merge_config_from_sysargv(sys.argv, c)
    common.run(c)
