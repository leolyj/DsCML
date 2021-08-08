# Sparse-to-dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation
This is the code related to "Sparse-to-dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation" (ICCV 2021).
<p align='center'>
  <img src='DsCML.jpg' width="1000px">
</p>

## Paper
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

You can follow the next steps to install the requairmented environment. This code is mainly modified from [xMUDA](https://github.com/valeoai/xmuda), you can also refer to its README if the installation isn't going well.

### 1 Setup a Conda environment:

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

### 2 Install nuscenes-devkit:

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
Using the cmd to finally install it:
```
pip install nuscenes-devkit
```
After the above steps, the devikit is installed, for any question you can refer to [devikit_installation_help](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/installation.md)



**If you meet the error with "pycocotools", you can try following steps:**

(1) Install Cython in your environment:
```
sudo apt-get installl Cython
```
```
pip install cython
```
(2) Download the [cocoapi](https://github.com/cocodataset/cocoapi) to your computer, decompress and enter it.

(3) Using cmd to enter the path under "PythonAPI", type:
```
make
```
(4) Type:
```
pip install pycocotools
```


### 3 Install SparseConveNet:

Download the [SparseConveNet](https://github.com/facebookresearch/SparseConvNet) to your computer, decompress, enter and develop it:
```
cd SparseConvNet/
bash develop.sh
```

## Datasets Preparation
For Dataset preprocessing, the code and steps are highly borrowed from [xMUDA](https://github.com/valeoai/xmuda), you can see more preprocessing details from this [Link](https://github.com/valeoai/xmuda). We summarize the preprocessing as follows:

### NuScenes
Download Nuscenes from [NuScenes website](https://www.nuscenes.org) and extract it.

Before training, you need to perform preprocessing to generate the data first. Please edit the script `DsCML/data/nuscenes/preprocess.py` as follows and then run it.

`root_dir` should point to the root directory of the NuScenes dataset

`out_dir` should point to the desired output directory to store the pickle files

### A2D2
Download the A2D2 Semantic Segmentation dataset and Sensor Configuration from the [Audi website](https://www.a2d2.audi/a2d2/en/download.html)

Similar to NuScenes preprocessing, please save all points that project into the front camera image as well as the segmentation labels to a pickle file.

Please edit the script `DsCML/data/a2d2/preprocess.py` as follows and then run it.

`root_dir` should point to the root directory of the A2D2 dataset

`out_dir` should point to the desired output directory to store the undistorted images and pickle files.

It should be set differently than the `root_dir` to prevent overwriting of images.

### SemanticKITTI
Download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip) from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract everything into the same folder.

Please edit the script `DsCML/data/semantic_kitti/preprocess.py` as follows and then run it.

`root_dir` should point to the root directory of the SemanticKITTI dataset
`out_dir` should point to the desired output directory to store the pickle files


## Usage

You can training the DsCML by using cmd or IDE such as Pycharm.
```
python xmuda/train_xmuda.py --cfg=../configs/nuscenes/day_night/xmuda.yaml
```
The output will be written to `/home/<user>/workspace` by  default. You can change the path `OUTPUT_DIR` in the config file in (e.g. `configs/nuscenes/day_night/xmuda.yaml`) 

You can start the trainings on the other UDA scenarios (USA/Singapore and A2D2/SemanticKITTI):
```
python xmuda/train_xmuda.py --cfg=../configs/nuscenes/usa_singapore/xmuda.yaml
python xmuda/train_xmuda.py --cfg=../configs/a2d2_semantic_kitti/xmuda.yaml
```

## Results
We present several qualitative results reported in our paper.




