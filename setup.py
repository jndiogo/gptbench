from setuptools import setup

setup(name='GPTbench',
      version='0.3.0',
      author='Jorge Diogo',
      packages=['gptbench'],
      description='A workbench to train, sample and measure GPT-2 level models',
      license='MIT',
      install_requires=[
            'numpy',
            'matplotlib',
            'torch',
            'tiktoken',
            'transformers' # to load pretrained GPT-2 models
      ],
)
