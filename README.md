# Installation 
To install Revive, version of nvidia driver should be greater than 440.  
We recommend to setup environment with Anaconda.
```
git clone https://agit.ai/Revive/revive.git
cd revive_core
conda create -n revive python=3.7
conda activate revive
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # install PyTorch

# install NeoRL
git clone https://agit.ai/Polixir/NeoRL
cd NeoRL
pip install -e .
cd ..

pip install -e . # install revive
```
To setup environment purely by pip, please make sure you install the correct version of PyTorch and CuPy along with CUDA.