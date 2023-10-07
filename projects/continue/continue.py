"""
Run a prompt with the GPT2 smallest model.

Other available models:
'gpt2': 124M params
'gpt2-medium': 350M params
'gpt2-large': 774M params
'gpt2-xl': 1558M params

To use another model, change init='...' below.

Another way of running, directly from commad line:

python -m gptbench.run -init=gpt2 -mode=prompt

"""

import sys

from gptbench import empty_config, config_run

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    c = empty_config()
    c.set(init='gpt2', mode='prompt')

    # train

    # sample

    # dataset
    c.dataset.class_name = 'gpt2'

    # model

    # trainer


    obj = config_run(c, sys.argv)