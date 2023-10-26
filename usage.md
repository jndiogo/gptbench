# Usage

To sample from a pretrained GPT2 model (the smaller one with 124M params):

```python
from gptbench import Sample, empty_config

ben = Sample('gpt2')

cfg = empty_config()
ben.init_pretrained('gpt2', cfg)

ben.sample("What a beautiful day")
```

The first time will take longer to download the pretrained model, which will then be cached locally.
Then, to enter an interactive prompt mode:

```python
ben.prompt()
```

Type -quit to exit, Ctrl+C to break during generation.


You can create a lightweight Sample object as above if you just want to sample from the model. 

To train it, a Train object must be created, then setup a dataset and a [config](config.md):

```python
from gptbench import Train, empty_config

ben = Train('shakespeare-model', seed=0xacac1a)

# set train dataset - no validation dataset to maximize training data
ben.set_datasets(class_name='gpt2', # GPT2TokensDataset
                 train_path='../data/shakespeare.txt', 
                 train_split=1.) 

# set config settings
cfg = empty_config()
cfg.model.set(n_layer=8, n_head=8, n_embd=128, block_size=64)
cfg.trainer.set(batch_size=128)

# init a blank model
ben.init_new(cfg)

# train it for 2000 iters
ben.train(iter_count=2000)

# sample something
ben.sample("So it goes")
```

Here we initialized a blank model with 8 layers, 8 heads, 128 dims for embedding and a block_size of 64. See the [Config](config.md) and [References](README.md#references) for help on these config settings.

During training, the model can be automatically saved every n steps, if the evaluated loss is better than before. It's saved with the name we gave during Train object creation - 'shakespeare-model'. This name can later be used to load the model checkpoint, in order to sampl efrom it or to continue training:

```python
# later, instead of ben.init_new(), we init by loading the checkpoint:
ben.load(name='shakespeare-model')

# the model can be manually saved at any time with:
ben.save(name='shakespeare-model')
```


## From Python script, Jupyter or command line

GPTBench can be used from python exactly as in the included Jupyter notebook examples. Another convenient way is by using the command line arguments to override default config options, which can be set in a script like this:

```python
import sys
from gptbench import config_run, empty_config

if __name__ == '__main__':

    # setup default config options
    c = empty_config()

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
```

The config_run() call in the last line will update the config with any -name=value entries you pass in the command line. For example, to override the trainer.batch_size to 256 you can call the above script with:

```
python script.py -init=new -mode=train -trainer.batch_size=256
```

In this manner, we can also init a new model, resume (load) a checkpoint for training or sampling, sample from it, etc - directly from the command line. You can break training at any time by interrupting the script with Ctrl+C, as the best checkpoint so far is periodically saved.

To have config_run() take care of everything, call the script with at least two args:

|Param|Values|
|-----|------|
|-init|new: same as init_new(), resume: load() an existing checkpoint, gpt2, gpt2-medium, gpt2-large, gpt2-xl: init from a pretrained GPT2 checkpoint, which will be downloaded from Hugging Face and cached locally.
|-mode|train: train model as specified in config, sample: sample from an initial text set in config.sample.start_text, prompt: run an interactive prompt where you can write start_text to feed the model's generation
|-name|You can also override any name set in the script used to load/save checkpoints. If not given, defaults to "model" and overwrites any existing with the same name|


Actually, you don't even need a script, as all options and config settings can be passed from the command line with python -m, for example:
```python
python -m gptbench.run -init=resume -mode=sample -sample.start_text="21+89="
```

So, the name, init and mode params choose what to do, while an argument like:
```python
-sample.start_text="21+89="
```

can be used to set config.sample.start_text with "21+89=" before running. See the add.py script in the add project and config_run() source for more options.




## Out-of-memory errors

Some larger models like gpt2-larger or gpt2-xl can be sampled from, but will require huge memory amounts to be trained.

If you get an out-of-memory error, try setting config trainer.batch_size to 1 and you might be able to train them with a 6Gb+ GPU, or slower with the CPU. Or better, you can use gradient accumulation to simulate large batch sizes with smaller ones.

To use gradient accumulation, set config trainer.accum_size to the micro batch size (that can fit in the GPU) and set trainer.batch_size to a multiple trainer.accum_size. In this manner GPTBench will accumulate gradients in batches of trainer.accum_size to simulate a larger trainer.batch_size batch.

