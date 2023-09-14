"""
"""

import sys

from gptbench.dataset import CharDataset
import gptbench.common as common

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    c = common.empty_config()

    # sample
    c.sample.eotstop = 1
    c.sample.multiline = 0

    # dataset
    c.dataset.cls = CharDataset

    # model

    # trainer



    c = common.merge_config_from_sysargv(sys.argv, c)
    common.run(c)