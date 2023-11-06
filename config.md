# Config

Because there are plenty of configurable parameters (model, training, dataset), everything runs based on a tree of config settings. You can set this config in many forms:

```python
cfg = empty_config()
cfg.model.block_size=256
cfg['model.block_size']=512
cfg.model.set(block_size=1024)
```

A config is passed when you initialize a model, by calling init_new(), init_pretrained() or load() and is used to override the default settings.

To see the current config being used, do:

```python
print(ben.get_config().dump(2))
```

You can get help about the avaliable settings by calling:

```python
ben.get_config().help()
```

Config is divided in these five areas:


## Sample
Sample config controls everything related to running the model in inference mode. This happens when you call sample() or during training if you enabled sampling logging with set_train_log_periods().
All settings below are accessed with sample.*, for example sample.max_len.

|Name | Info |
|-----|------|
|max_len|Max generated token count|
|count|How many times to generate from the same start_text|
|start_text|Starting text for generation. None: use random vocabulary item on each sampling. A str with starting text.If separated with start_text_sep, multiple star_text are used (config.count is set to 1)|
|start_text_sep|When used in start_text, this char separates multiple start strings. Default is pipe character '\|'|
|emit_start|When sampling, emit start_text? Only if emit_after is None|
|emit_after|When sampling, only emit after this text has been seen|
|emit_before|When sampling, stop before emitting this. If both emit_before and emit_until are set, emit_before is used. With flush=1 only works for single chars|
|emit_until|When sampling, stop after emitting this, but still display it. If both emit_before and emit_until are set, emit_until is used. With flush=1 only works for single chars|
|flush|When sampling, should each token display immediately?|
|eot_stop|Should generation stop when dataset's special End-Of-Text token is emitted? 0=don't stop, -1=stop before, 1=stop after (and display it)|
|top|Top_k or top_p (nucleus) filtering: 0: off (default),  1: most probable,  ]0..1[: top_p(n) sampling,  >1: top_k(int(n)) sampling,  [-1..0[: top_k(vocab_size * -n) (vocab_size -ratio). Examples - to keep only the most probable entry: 1 (top_k), to sample from 40 most probable: 40 (top_k), to sample from top 60% of accumulated probabilities: 0.6 (top_p)|
|temp|Temperature|
|max_batch_size|Maximum batch size when inferring in parallel with multiple start text. None means same as trainer.batch_size config entry|
|multiline_prompt|In prompt() mode only: input multiple lines until a Ctrl+D (or Ctrl+Z in Windows). Doesn't work in Jupyter notebooks|


## Train
Train config section controls evaluation options, when you call the train() method. See also the Trainer section below.

|Name | Info |
|-----|------|
|eval_period|In batch iterations: each n batches we eval and check if saving model. 0 for none|
|eval_type|How to estimate loss -> 0: on train data, 1: on val data (or train if no val dataset), ]0,1[: weighted average of train and val (or train only if no val dataset|
|eval_iters|Count of batch_size iterations for loss evaluation|
|eval_save_checkpt|When to save a checkpoint: 0=never, 1=on new lower evaluated loss, 2=always|
|eval_save_loss|Loss logging into checkpoint's logs folder. Multiple values allowed, separated with comma: 'csv' saves a loss.csv, 'plot' plots a loss chart to loss.png, 'tensorboard' creates tensorboard logs|


## Dataset
Configuration for the dataset:

|Name | Info |
|-----|------|
|class_name|'gpt2': dataset using the GPT2 tokenizer from the tiktoken library, 'char': a utf-8 character-based dataset with samples served continuously, 'charline': a character-based utf-8 dataset read from a line-based text file where each line is a sample|
|train_path|Train dataset path|
|train_split|Train dataset split ratio (0..1) for creating a validation dataset. Only used if val_path is unset|
|val_path|, Validation dataset path. If set, train_split is not used|
|params|String in the form 'name1=value1,name2=value2,...' containing extra parameters for dataset creation|


## Model
Model controls the GPT2 model settings:

|Name | Info |
|-----|------|
|device|Device for running the net: 'cuda', 'cpu' or any other supported by Torch. 'auto' for CUDA or CPU if no CUDA|
|dtype|Data type: 'float32', 'bfloat16'|
|n_layer|Number of transformer blocks|
|n_head|Number of attention heads|
|n_embd|Number of embedding dimensions. Must be a multiple of n_head|
|vocab_size|Size of the vocabulary. Must be set from dataset in use|
|block_size|The number of vocabulary items processed at a time. If set before loading or initializing from a pretrained model, will crop the loaded block_size|
|dropout|Dropout hyperparameter|
|flash_attn|Use the more efficient Flash Attention method, on torch 2+ and supporting devices|


## Trainer
The trainer config section controls optimizer and other training parameters:

|Name | Info |
|-----|------|
|batch_size|Size of the batch in each forward training iteration|
|accum_size|Size for batch gradient accumulation, allowing for larger batch sizes with lower memory usage. Setting batch_size must be a multiple of accum_size. For example: batch_size=32, accum_size=4 will simulate a batch of 32 by accumulating gradients on 8 batches of 4 rows|
|n_workers|DataLoader workers. In Windows setting to greater than 0 causes a long delay when calling iter().|
|max_samples|Absolute maximum limit on training samples. Negative -n for number of epochs|
|grad_norm_clip|Clip gradients to this norm|
|optimizer|Optimizer type: 'sgd' or 'adamw'|
|learning_rate|Initial learning rate|
|adamw_beta1|AdamW beta1|
|adamw_beta2|AdamW beta2|
|adamw_weight_decay|AdamW weight decay, only applied on matmul weights|

Other info about config settings, like default values or data types, can be gathered from the source code, for example the model config is initialized in model.py.

