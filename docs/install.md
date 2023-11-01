# INSTAllation 
## Requirements
* Linux **(Recommend)**
* Python 3.7+ 
* PyTorch ≥ 1.7 
* CUDA 9.0 or higher

I have tested the following versions of OS and softwares：
* OS：Ubuntu 18.04
* CUDA: 10.0/11.3

## Install 
**CUDA Driver Version ≥ CUDA Toolkit Version(runtime version) = torch.version.cuda**

a. Create a conda virtual environment and activate it, e.g.,
```
conda create -n Py37_Torch1.10_cu11.3 python=3.7 -y 
source activate Py37_Torch1.10_cu11.3
```
b. Make sure your CUDA runtime api version ≤ CUDA driver version. (for example 11.3 ≤ 11.4)
```
nvcc -V
nvidia-smi
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), Make sure cudatoolkit version same as CUDA runtime api version, e.g.,
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
nvcc -V
python
>>> import torch
>>> torch.version.cuda
>>> exit()
```
d. Clone the RotatedRiceSpikeDet repository.
```
git clone https://github.com/tinly00/RotatedRiceSpikeDet.git
cd RotatedRiceSpikeDet
```
e. Install RotatedRiceSpikeDet.

```python 
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```

## Install DOTA_devkit. 
**(Custom Install, it's just a tool to split the high resolution image and evaluation the obb)**

```
cd yolov5_obb/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

