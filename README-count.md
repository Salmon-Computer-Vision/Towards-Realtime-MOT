# count.py

# ARM64 Jetson Nano Setup

To install `numba`, `llvm-9` must be installed.

```bash
sudo apt install llvm-9
sudo ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config
```

Setup and build tools for the packages:
```bash
pip3 install -U pip
pip3 install setuptools wheel
```

[PyTorch
1.10.0](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) for
Python 3.6 which is the only one supported in the latest Jetpack for Jetson
Nano:

```bash
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install Cython
pip3 install numpy torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

[Torchvision 0.11.0](https://qengineering.eu/install-pytorch-on-jetson-nano.html)
```bash
# the dependencies
$ sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo pip3 install -U pillow
# install gdown to download from Google drive, if not done yet
$ sudo -H pip3 install gdown
# download TorchVision 0.11.0
$ gdown https://drive.google.com/uc?id=1C7y6VSIBkmL2RQnVy8xF9cAnrrpJiJ-K
# install TorchVision 0.11.0
$ sudo -H pip3 install torchvision-0.11.0a0+fa347eb-cp36-cp36m-linux_aarch64.whl
```

# Usage

Run count for entire folder of videos:
```bash
python count.py --output-format none --cfg weights/cfg/yolo3.cfg --weights weights/latest.pt recursive path/to/vid_folder
```

Parse countable csv files in folder:
```bash
python count.py count countables/
```

# Caveats

Install each package one at a time:
```bash
xargs -L 1 pip install < requirements.txt
```

May need to upgrade numpy if there is an error:
```bash
pip install -U numpy
```

Upgrade pytorch for your CUDA distribution (Eg. CUDA 11.3):
```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
