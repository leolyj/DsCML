# Sparse-to-dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation
This is the code related to "Sparse-to-dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation" (ICCV 2021).
<p align='center'>
  <img src='DsCML.jpg' width="1000px">
</p>

# Paper
[Paper Link](https://arxiv.org/abs/2107.14724)  
IEEE International Conference on Computer Vision (ICCV 2021)

If you find it helpful to your research, please cite as follows:

```
@inproceedings{peng2021sparse,
  title={Sparse-to-dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation},
  author={Peng, Duo and Lei, Yinjie and Li, Wen and Zhang, Pingping and Guo, Yulan},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021},
  publisher={IEEE}
}
```

## Preparation
* PyTorch 1.7.1
* CUDA 11.1
* Python 3.7.9
* Torchvision 0.8.2
* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

You can follow the next steps to install the requairmented environment. This code is mainly modified from [**xMUDA**](https://github.com/valeoai/xmuda), you can also refer to its README if the installation isn't going well.

1 Setup a Conda environment:
We create a new Conda environment named `nuscenes`. We will use this environment for both nuScenes and nuImages.
```
conda create --name nuscenes python=3.7
```
You can enable the virtual environment using:
```
conda activate nuscenes 
```
To deactivate the virtual environment, use:
```
source deactivate
```

2 Install nuscenes-devkit:
Download the [devkit](https://github.com/nutonomy/nuscenes-devkit) to your computer, decompress and enter it.

Add the `python-sdk` directory to your `PYTHONPATH` environmental variable, by adding the following to your `~/.bashrc` :
```
export PYTHONPATH="${PYTHONPATH}:$HOME/nuscenes-devkit/python-sdk"
```
Using cmd (make sure the environment "nuscenes" is activated) to install the base environment:
```
pip install -r setup/requirements.txt
```
Setup environment variable:
```
export NUSCENES="/data/sets/nuscenes"
```
After the above steps, the devikit is installed, for any question you can refer to [devikit_installation_help](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/installation.md)






* Download [**Painter by Numbers**](https://www.kaggle.com/c/painter-by-numbers/) which are paintings for GTR.
* You should transfer the raw source dataset into multiple-style datasets using the pre-trained style transfer network [AdaIN](https://github.com/xunhuang1995/AdaIN-style) and put the correct paths in line 685 of the python file (./tools/TR_BR.py )
* Download [**the model**](http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth) pretrained on ImageNet. Put it into each file named as  (pretianed_model).
* This code are mainly modified from [**xMUDA**](https://github.com/valeoai/xmuda), please refer to its readme.md if more details are wondered.

## Usage

Open the terminal and type the following command to pretrain the model on the source domain (GTA5).
```
python3 tools/TR_BR.py
```

## Results
We present several qualitative results reported in our paper.




