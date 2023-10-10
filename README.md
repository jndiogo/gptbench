# GPT Bench

GPTBench is a workbench where you can experiment with transformer models like GPT.

It aims at a more "intimate" contact with GPT-like transformer models. What can transformers learn even without giga-data (and still fitting our GPUs)? "Small is beautiful"

GPTBench is a kind of lab or workbench because you can use it to conveniently train a large or small transformer model and see what it can learn.

Model sampling is simple and also includes a prompt mode where you can continuously interact with the model without having to reload checkpoints. You can train it starting from a blank model or from a pretrained GPT2. Checkpoints can be loaded and saved at any point. Can measure accuracy and can log training evaluations to a .csv and TensorBoard formats. The gptbench package can be used in Python scripts or Jupyter notebooks or directly from the command line.

Includes simple examples for character-level and GPT2 token level datasets, for things like learning to add two numbers, decimal-roman numeral translation, number sequences and of course English text completion.

This package grew from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT), to whom I must express my gratitude for the many and [very inspiring lessons](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). The model.py, trainer.py are mostly the same, adapted to be integrated with the "workbench".

Hope this can be a contribution to a better understanding of these weird and fascinating machines, the transformers.




## Installation

Requires Python 3.7+ and PyTorch. Also uses NumPy, the tiktoken library (for the BPE tokenizer) and Hugging Face's transformers library (just to download GPT2 checkpoints).

You can run it in plain CPU or a CUDA GPU.

To use the gptbench package, download the repository and from the base directory (which has a setup.py script) do:

```
pip install -e .
```


## Projects

See [Projects](projects/README.md).




## Usage

To sample form the GPT2 smaller model (124M params):

```python
from gptbench import Sample, Train, empty_config

ben = Sample('gpt2')

cfg = empty_config()
ben.init_pretrained('gpt2', cfg)

ben.sample("What a beautiful day")
```


To enter an interactive prompt mode:

```python
ben.prompt()
```

Type -quit to exit, Ctrl+C to break during generation.


You can create a lightweight Sample object as above if you just want to sample from the model. 

To train it, a Train object must be created, then setup a config and a dataset:


```python
ben = Train('shakespeare', seed=0xacac1a)

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

Here we initialized a blank model with 8 layers, 8 heads, 128 dims for embedding and a block_size of 64. See the [Config](#config) and [References](#references) below for help on these config settings.

After training, a model can be saved and later loaded:

```python
ben.save(name='skp-today')

# at a later time, in place of the above ben.init_new()...
ben.load(name='skp1-today')
```

### From Python script, Jupyter or command line

GPTBench can be used from a python script exactly as in the included Jupyter notebook examples. Another way is by using the command line arguments to override default config options, which can be set in a script like this:

```python

import sys

from gptbench import Train, empty_config, config_run

# -----------------------------------------------------------------------------

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

The config_run() call in the last line will update the config with any name=value entries you pass in the command line. For example to override the trainer.batch_size to 256 you can call the script with:

```
python script.py -init=new -mode=train -trainer.batch_size=256
```

In this manner, we can also init a new model, resume (load a checkpoint) for training or sampling, sample from it, etc - and all this from the command line.

To run, call the script with at least two args:

|Param|Values|
|-----|------|
|-init|new: same as init_new(), resume: load() an existing checkpoint, gpt2, gpt2-medium, gpt2-large, gpt2-xl: init from a pretrained GPT2 checkpoint, which will be downloaded from Hugging Face and cached locally.
|-mode|train: train model as specified in config.train, model, trainer and dataset, sample: sample from an initial text in config.sample.start_text, according to other config.sample settings, prompt: run an interactive prompt where you can write start_text to feed the model's generation
|-name|You can also override any name set in the script used to load/save checkpoints|

See the add.py script in the add project and config_run() for more options.


Actually, you don't even need a script, as all options and config settings can be passed from the command line, for example:

python -m gptbench.run -name=add2_script -init=resume -mode=sample -sample.start_text="21+89="

So, the name, init and mode choose what to do, while an argument like:

-sample.start_text="21+89="

can be used to set config.sample.start_text with "21+89=" before running.








### Out-of-memory errors

Some larger models like gpt2-* can be sampled from, but will require huge memory amounts to be trained. If you get an out-of-memory error, set config.trainer.batch_size to 1 and you might be able to train them with a 6Gb GPU, or slower with the CPU. 

## Config

Because there are plenty of configurable parameters (model, training, dataset), everything runs based on a settings tree called config. You can set this config in many forms:

```python
cfg = empty_config()
cfg.model.block_size=256
cfg['model.block_size']=512
cfg.model.set(block_size=1024)
```

A config is passed when you initialize a model, by calling init_new(), load() or init_pretrained() and is used to override settings as needed. For more information on how config settings are used, please check the source code.

Config is divided in these five areas:

### Sample
Sample config controls everything related to running the model in inference mode. This happens when you call sample() or during training if you enabled sampling with set_train_log_periods().
All the above are accessed with sample.*.

|Name | Info |
|-----|------|
|max_len|Max generated token count|
|count|How many times to generate from the same start_text|
|start_text|Starting text for generation. None: use random vocabulary item on each sampling. A str with starting text.If separated with start_text_sep multiple star_text are used (count is set to 1)|

|start_text_sep|When used in start_text, this char separates multiple start strings. Default is pipe character '\|'|
|emit_start|When sampling, emit start_text? Only if emit_after is None|
|emit_after|When sampling, only emit after this text has been seen|
|emit_before|When sampling, stop before emitting this. With flush=1 only works for single chars|
|flush|When sampling, should each token display immediately?|
|eot_stop|Should generation stop when dataset's special End-Of-Text token is emitted? 0=don't stop, -1=stop before, 1=stop after (and display it)|
|top|Top_k or top_p filtering: 0: off,  ]0..1]: top_p,  [-1..0[: top_k(vocab_size * -top),  >=1: top_k(int(n))|
|temp|Temperature|
|max_batch_size|Maximum batch size when inferring in parallel with multiple start text. None means no limit which produce out-of-memory errors in larger models|
|multiline_prompt|On prompt mode: input multiple lines until a Ctrl+D or Ctrl+Z (in Windows)|



### Train
Train config section controls evaluation options, when you call the train() method. See also the Trainer section below.

|Name | Info |
|-----|------|
eval_period|In batch iterations: each n batches we eval and check if saving model. 0 for none|
|eval_type|How to estimate loss -> 0: on train data, 1: on val data (or train if no val dataset), ]0,1[: weighted average of train and val (or train only if no val dataset|
|eval_iters|Count of batch_size iterations for loss evaluation|
|eval_save_checkpt|When to save a checkpoint: 0=never, 1=on new lower evaluated loss, 2=always|
|eval_save_loss|Multiple values allowed, separated with comma: 'csv' saves a loss.csv, 'tensorboard' creates tensorboard logs|

### Dataset
Configuration for the dataset:

|Name | Info |
|-----|------|
|class_name|'gpt2': dataset using the GPT2 tokenizer from the tiktoken library, 'char': a utf-8 character-based dataset with samples served continuously, 'charline': a character-based utf-8 dataset read from a line-based text file where each line is a sample|
|train_path|Train dataset path|
|train_split|Train dataset split ratio (0..1) for creating a validation dataset. Only used if val_path is unset|
|val_path|, Validation dataset path. If set, train_split is not used|
|params|String in the form 'name1=value1,name2=value2,...' containing extra parameters for dataset creation|

### Model
Model controls the GPT2 model settings:

|Name | Info |
|-----|------|
|device|Device for running the net: 'cuda', 'cpu' or any other supported by Torch. 'auto' for CUDA or CPU if no CUDA|
|dtype|Data type: 'float32', 'bfloat16'|
|n_layer|Number of transformer blocks|
|n_head|Number of attention heads|
|n_embd|Number of embedding dimensions. Must be a multiple of n_head|
|vocab_size|Size of the vocabulary. Must be set from dataset in use|
|block_size|Block size: number of vocabulary items processed at a time. Must be set|
|dropout|Dropout hyperparameter|


### Trainer
The trainer config section controls optimizer and other training parameters:

|Name | Info |
|-----|------|
|n_workers|DataLoader workers. In Windows setting to above 0 causes a long delay when calling iter().|
|batch_size|Size of the batch in each forward training iteration|
|max_samples|Absolute maximum limit on training samples. Negative -n for number of epochs|
|grad_norm_clip|Clip gradients to this norm|
|optimizer|Optimizer type: 'sgd' or 'adamw'|
|learning_rate|Initial learning rate|
|adamw_beta1|AdamW beta1|
|adamw_beta2|AdamW beta2|
|adamw_weight_decay|AdamW weight decay, only applied on matmul weights|

Other info about config settings, like default values can be gathered from the source.




## Todo

- Document dataset classes and that users can pass their own.
- Add better examples of GPT2 fine tuning.
- Add shared batch/gradient accumulation to allow larger batch sizes. Available memory measurement will be needed.
- Add Conf.help to list all help information in the registry


## References
- [Weird world of LLMs](https://simonwillison.net/2023/Aug/3/weird-world-of-llms/)
- [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT)
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)




### License
MIT
