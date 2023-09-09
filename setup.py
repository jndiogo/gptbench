from setuptools import setup

setup(name='GPTbench',
      version='0.0.1',
      author='Jorge Diogo',
      packages=['gptbench'],
      description='A workbench to train and sample from GPT-2 level models. Based on Andrej Karpathy\'s minGPT',
      license='MIT',
      install_requires=[
            'numpy',
            'torch',
            'tiktoken',
            'transformers' # to load pretrained GPT-2 models
      ],
)
