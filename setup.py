import os
from setuptools import setup
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'revive_core'))
from version import __version__

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

setup(
    name='revive',
    author='Icarus@Polixir',
    author_email="wizardicarus@gmail.com",
    py_modules=['revive_core'],
    version=__version__,
    install_requires=[
        'dataclasses==0.6',
        'torch>=1.6',
        'torchvision',
        'ray==1.1',
        'lightgbm==2.3.1',
        'scikit_learn==0.23.2',
        'pyro-ppl',
        'tabulate',
        'tensorboardX',
        'numpy',
        'scipy',
        'matplotlib',
        'tensorboard',
        'tqdm',
        'pot',
        'pandas',
        'tianshou',
        'zoopt>=0.4.1',
        'bayesian-optimization'
    ],
    url="https://agit.ai/Polixir/revive"
)
