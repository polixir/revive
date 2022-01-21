import os
import sys
from setuptools import setup, find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'revive'))
from version import __version__

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

setup(
    name='revive',
    author='Polixir Technologies Co., Ltd.',
    py_modules=['revive'],
    packages=find_packages(),
    version=__version__,
    install_requires=[
        'aiohttp==3.7.4',
        'dataclasses==0.6',
        'torch>=1.8',
        'pyro-ppl==1.6.0',
        'aioredis==1.3.1',
        'ray==1.8',
        'loguru',
        'tabulate',
        'tensorboardX',
        'numpy<=1.20',
        'scipy',
        'matplotlib',
        'tensorboard',
        'gym',
        'tqdm',
        'pot',
        'pandas',
        'zoopt>=0.4.1',
        'h5py',
        'pytest',
        'CairoSVG==2.5.2',
        'dtreeviz==1.3.2',
        'PyPDF2==1.26.0',
        'pyarmor==7.3.0',
    ],
    url="https://agit.ai/Polixir/revive"
)
