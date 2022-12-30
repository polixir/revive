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
        'torch>=1.12',
        'pyro-ppl==1.8.2',
        'ray[default]==2.1.0',
        'bayesian-optimization',
        'loguru',
        'tabulate',
        'tensorboardX',
        'numpy',
        'pyarrow',
        'scipy',
        'matplotlib',
        'tensorboard',
        'gym==0.25.0',
        'tqdm',
        'pot',
        'pandas',
        'zoopt==0.4.2',
        'h5py',
        'pytest',
        'CairoSVG==2.5.2',
        'dtreeviz==1.4.0',
        'PyPDF2==1.26.0',
        'pyarmor==7.3.0',
        'prometheus_client==0.13.1',
        'prettytable',
        'wurlitzer',
        'causal-learn==0.1.3.0',
    ],
    url="https://agit.ai/Polixir/revive"
)

config_folder = os.path.join(os.path.expanduser('~'),".revive")
if not os.path.exists(config_folder):
    os.makedirs(config_folder)
    with open(os.path.join(config_folder,'config.yaml'), 'w', encoding='utf-8') as f:
        lines = []
        lines.append("accesskey: xxxxxxxxx"+"\n")
        f.writelines(lines)