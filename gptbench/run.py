"""

"""

import os, sys, copy, datetime

from .sample import Sample
from .train import Train

from .config import merge_config_from_sysargv, empty_config

from .conf import Conf


# -----------------------------------------------------------------------------
def config_run(config, sys_argv = None):

    if sys_argv is not None:
        config = merge_config_from_sysargv(sys.argv, config)

    assert config.has('init') and config.has('mode'), "Please provide at least the init=new|resume|gpt2* and mode=train|sample|prompt params"

    mode=config.mode
    del config.mode
    init = config.init
    del config.init


    # fill-in optional object creation params and remove them from config
    ops = {}
    if config.has('name'):
        ops['name'] = config.name
        del config.name # delete from config because these are not real config entries

    if config.has('work_dir'):
        ops['work_dir'] = config.work_dir
        del config.work_dir

    # logs
    if config.has('log_mask'):
        ops['log_mask'] = int(config.log_mask)
        del config.log_mask

    # only for training
    if config.has('log_dot_period'):
        if mode == 'train':
            ops['log_dot_period'] = float(config.log_dot_period)
        del config.log_dot_period
    if config.has('log_loss_period'):
        if mode == 'train':
            ops['log_loss_period'] = float(config.log_loss_period)
        del config.log_loss_period
    if config.has('log_sample_period'):
        if mode == 'train':
            ops['log_sample_period'] = float(config.log_sample_period)
        del config.log_sample_period


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
        do.load(config)

    else:
        assert False, f"Unknown init '{init}'"


    if sys_argv is not None: # save argv into run.log
        do.ensure_path()

        text = ''

        if init != 'resume':
            text += 'Initial config:\n' + do.config.dump(1) + '\n\n'

        now = datetime.datetime.now()

        text += now.strftime("%Y/%m/%d %H:%M:%S") + ':\n' + ' '.join(sys_argv) + '\n\n'

        do.path_append('run.log', text, clear=init != 'resume')


    # train or sample
    if mode == 'train':
        do.train()
    elif mode == 'sample':
        if config.sample.has('start_text'):
            do.sample(config.sample.start_text)
        else:
            raise ValueError('sample.start_text must be given')

    elif mode == 'prompt':
        do.prompt()

    return do




if __name__ == '__main__':

    c = empty_config()
    obj = config_run(c, sys.argv)
