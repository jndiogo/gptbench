"""

"""

import os, sys, copy

from .sample import Sample
from .train import Train

from .config import merge_config_from_sysargv

from .utils import CfgNode


# -----------------------------------------------------------------------------
def config_run(config, sys_argv = None):

    if sys_argv is not None:
        config = merge_config_from_sysargv(sys.argv, config)

    assert config.has('mode') and config.has('init'), "Config must include config.mode and config.init"


    # fill-in optional object creation params
    ops = {}
    if config.has('name'):
        ops['name'] = config.name
    if config.has('log_mask'):
        ops['log_mask'] = config.log_mask
    if config.has('work_dir'):
        ops['work_dir'] = config.work_dir

    if config.mode == 'train':
        do = Train(**ops)
    elif config.mode == 'sample' or config.mode == 'prompt':
        do = Sample(**ops)
    else:
        assert False, f"Unknown mode '{config.mode}'"

    # init
    if config.init == 'new':
        do.init_new(config)

    elif config.init.startswith('gpt2'):
        do.init_pretrained(config.init, config)

    elif config.init == 'resume':
        do.resume(config)

    else:
        assert False, f"Unknown init '{config.init}'"

    # train or sample
    if config.mode == 'train':
        do.train()
    elif config.mode == 'sample':
        do.sample()
    elif config.mode == 'prompt':
        do.prompt()

    return do
