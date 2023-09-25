""" GPT Bench """

__version__ = '0.0.1'

__all__ = [
    'Sample', 'LogFlag',
    'Train',
    'GPT2TokensDataset', 'CharDataset', 'PaddedLineCharDataset',
    'empty_config', 'config_run', 'merge_config_from_sysargv'
]

__author__ = 'Jorge Diogo'

from .sample import Sample
from .train import Train

from .tokendataset import GPT2TokensDataset
from .chardataset import CharDataset, PaddedLineCharDataset

from .config import empty_config, merge_config_from_sysargv, LogFlag
from .run import config_run

from .utils import consumme_decode_utf8
