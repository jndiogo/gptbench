"""

"""

import os, sys, copy

from .sample import Sample
from .train import Train

from .config import merge_config_from_sysargv

from .conf import Conf


# -----------------------------------------------------------------------------
def config_run(config, sys_argv = None):

    if sys_argv is not None:
        config = merge_config_from_sysargv(sys.argv, config)

    assert config.has('mode') and config.has('init'), "Please provide at least the mode and init params"


    # fill-in optional object creation params
    ops = {}
    if config.has('name'):
        ops['name'] = config.name
        del config.name # delete from config because these are not real config entries
    if config.has('work_dir'):
        ops['work_dir'] = config.work_dir
        del config.work_dir
    if config.has('log_mask'):
        ops['log_mask'] = int(config.log_mask)
        del config.log_mask

    mode=config.mode
    del config.mode
    init = config.init
    del config.init


    if mode == 'train':
        do = Train(**ops)
    elif mode == 'sample' or mode == 'prompt':
        do = Sample(**ops)
    else:
        assert False, f"Unknown mode '{mode}'"

    # init
    if init == 'new':
        do.init_new(config)

    elif init.startswith('gpt2'):
        do.init_pretrained(init, config)

    elif init == 'resume':
        do.init_resume(config)

    else:
        assert False, f"Unknown init '{init}'"


    if sys_argv is not None and init != 'resume': # save argv cmd
        do.ensure_path()        
        cmd = ' '.join(sys_argv)
        text = cmd + '\n\nConfig:\n' + str(do.config)
        do.path_save('init.txt', text)

    # train or sample
    if mode == 'train':
        do.train()
    elif mode == 'sample':
        if config.sample.has('start_text'):
            do.sample(config.sample.start_text)
        else:
            raise Exception('sample.start_text must be given')

    elif mode == 'prompt':
        do.prompt()

    return do
