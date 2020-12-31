# FDTD_SiliconWG
## Introduction
本程序利用2d-FDTD with PML方法对Silicon waveguide进行仿真，使用语言为Python. 如果没有安装Python的话，这里首先建议使用环境管理器 [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)安装Python，版本为3.7+。
## How to run this repo

使用方法，这里默认已经安装好Anaconda or Miniconda：

1. 首先将代码克隆到本地：``git clone git@github.com:dantehustg/FDTD_SiliconWG.git``
2. 克隆好以后运行命令(Linux or Window Powershell): ``cd FDTD_SiliconWG``
3. 创建运行的虚拟环境：``conda create --name fdtd_wg python=3.7``
4. 激活虚拟环境：``conda activate fdtd_wg``
5. 安装运行所需的包：``pip install -r requirements.txt``
6. 打开python交互命令行，输入``import numpy, matplotlib``，没有报错，即说明环境配置成功。
7. 在终端输入：``python ./waveguide/slabwg.py`` 运行demo, 这可能需要运行1-2分钟。

代码运行的结果会在``results/slab_waveguide``中，包括最终的gif以及中间时刻的场。 ``example``文件夹中有[Lumerical](https://www.lumerical.com/)和本模拟的对比图。
## Futher Work (TODO)
- Modularization.
- More simulation on different devices (e.g. photonic crystal waveguide).

## Auther Contribution
**胡琪鑫 (Qixin hu)**: All this repo, including main FDTD update program, perfectly matched layers (PML), source, device, visualization and README.

**冯德龙 (Delong Feng)**: [Lumerical](https://www.lumerical.com/) simulation and necessary discussion on FDTD update equation with PML. Add TF/SF source on [1dfdtd](./waveguide/1dfdtd.py).