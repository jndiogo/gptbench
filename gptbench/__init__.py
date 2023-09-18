# https://github.com/python/cpython/blob/3.6/Lib/json/__init__.py

__version__ = '0.0.1'

__all__ = [
    'Sample', 
    'LogFlag',
    'Train'
]

from .sample import Sample
from .train import Train
from .config import LogFlag
