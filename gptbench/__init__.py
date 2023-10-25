""" GPT Bench """

__version__ = '0.0.1'

__all__ = [
    'Sample', 'LogFlag',
    'Train',
    'GPT2TokensDataset', 'CharDataset', 'CharLineDataset',
    'empty_config', 'Conf', 'merge_config_from_sysargv',
    'GPT'
]

__author__ = 'Jorge Diogo'

from .sample import Sample
from .train import Train

from .tokendataset import GPT2TokensDataset
from .chardataset import CharDataset, CharLineDataset

from .config import empty_config, merge_config_from_sysargv, LogFlag
from .conf import Conf

from .model import GPT

from .utils import consumme_decode_utf8



"""
Deal with warning when running with -m gptbench.run:
RuntimeWarning: 'gptbench.run' found in sys.modules after import of package 'gptbench', but prior to execution of 'gptbench.run'; this may result in unpredictable behaviour

A bit of a hack, but reasonable and contained.
"""
import sys
if not '-m' in sys.argv:
	from .run import config_run
	__all__.append('config_run')