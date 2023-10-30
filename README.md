# GPTBench

This is a workbench where you can experiment with transformer LLM models. It aims at allowing a more "intimate" contact with GPT-like transformer models. What can transformers learn even without giga-data (and hopefully still fitting a GPU)?

It's through size that LLMs achieve amazing results, but size is a big barrier for normal people to play with them. The GPT-2 level models are on the border of what can be trained on commonly available GPUs and can thus be very valuable. They're not the smartest, but they are quite pliable.

GPTBench can be used to conveniently train a large or small transformer model and see what it can learn. It's made for tinkering and learning.

- Model sampling is simple and also includes a prompt mode where you can continuously interact with the model without having to reload checkpoints. 
- You can train it starting from a blank model or from a pretrained GPT-2. Checkpoints can be loaded and saved at any point.
- Can measure accuracy, perplexity and loss metrics and can log training evaluations to .csv and TensorBoard formats.
- The gptbench package can be used in Python scripts or Jupyter notebooks or directly from the command line.
- Includes simple examples for character-level and GPT2 token level datasets, for things like learning to add two numbers, decimal-roman numeral translation, number sequences and of course English text completion.
- Can train with gradient accumulation, so that larger batch sizes can be used in common GPUs.

This package grew from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT), to whom I'm very grateful for the [very inspiring lessons](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). The classes in model.py and trainer.py are mostly the same as in minGPT, adapted to be integrated in this "workbench".

Hoping this can be a contribution to a better understanding of these weird and fascinating machines, the decoding transformers.


## Installation

Requires Python 3.7+ and PyTorch 2. Also uses NumPy, the tiktoken library (for the BPE tokenizer) and Hugging Face's transformers library (to download GPT-2 checkpoints).

You can run it in a plain CPU or CUDA GPU.

To use the gptbench package, download the repository and from the base directory (which has a setup.py script) do:

```
pip install -e .
```


## Projects

Take a look at the [Projects](projects/readme.md) to see examples of what you can do with it.



## Usage

For examples, see [Usage](usage.md) and the Jupyter notebooks and python scripts in [Projects](projects/readme.md).

The way to select options in GPTBench is via settings in a config tree. See [here](config.md) for all the config options.



## Todo

- Add examples of GPT2 fine tuning.
- Document dataset classes and how users can pass their own. As dataset classes become consolidated, derive from a base.



## References

Good info about transformers and Large Language Models:

- [Weird world of LLMs](https://simonwillison.net/2023/Aug/3/weird-world-of-llms/)
- [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT)
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)



### License
MIT
