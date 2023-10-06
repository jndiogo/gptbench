"""
GPTBench can be used from a python script exactly as in the jupyter notebook examples.

Another way is by using the command line arguments to override config options set in a script like this.

In this manner, we can init a new model, resume (load a checkpoint) for training or sampling, sample from it, etc
- and all this from the command line.

To run, call the script with at least two args:

-init=new: same as init_new()
      resume: load() an existing checkpoint
      gpt2, gpt2-medium, gpt2-large, gpt2-xl: init from a pretrained GPT2 checkpoint, which wil be downloaded from Hugging Face and cached locally.

-mode=train: train model as specified in config.train, model, trainer and dataset
      sample: sample from an initial text in config.sample.start_text, according to other config.sample settings
      prompt: run an interactive prompt where you can write start_text to feed the model's generation

-name=you can also override the name set in the script used to load/save checkpoints


See config_run() for more options.


Actually, you don't even need a script, as all options and config settings can be passed from the comand line, for example:

python -m gptbench.run -name=add2_script -init=resume -mode=sample -sample.start_text="21+89="

The name, init and mode choose what to do, while an argument like:
-sample.start_text="21+89="

can be used to set config.sample.start_text with "21+89=" before running.

See the config help for the available settings.

"""

import sys

from gptbench import Train, empty_config, config_run

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # setup config options
    c = empty_config()

    c.name='add2_script'
    c.seed=0xB055ADD2 # 0 for random seed
    c.log_loss_period=0

    # train
    c.train.eval_period=100

    # sample
    c.sample.set(top=1) # top_k(1) - always pick the best item

    # dataset
    c.dataset.set(class_name='charline',
                  train_path='../data/add2.txt', 
                  train_split=0.9,
                  params='pre_shuffle=True')

    # model
    c.model.set(n_layer=6, 
                n_head=6, 
                n_embd=90, 
                block_size=16)

    # trainer
    c.trainer.set(batch_size=128)


    obj = config_run(c, sys.argv)
