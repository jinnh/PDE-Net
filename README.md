PDE-Net
======
**This is an implementation of Deep Posterior Distribution-based Embedding for Hyperspectral Image Super-resolution.**

[[arXiv]](https://arxiv.org/abs/2205.14887 "arXiv"), [[IEEE]](https://ieeexplore.ieee.org/document/9870666 "IEEE")

Requirement
---------
**python 3.7, Pytorch 1.7.0, and cuda 11.0**

Quick Start
--------
### Dataset

You can refer to the following links to download the datasets, [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), and [Harvard](http://vision.seas.harvard.edu/hyperspec/ "Harvard"). And run the matlab programs in the folder 'datasets' to get the pre-processed training and testing data.

### Training

**You can train directly by using the file 'train.sh':**

	bash train.sh

**Or you can execute the following commands respectively:**
	
	python train.py --cuda --gpu "0" --dataset "CAVE" --upscale_factor 4 --model_name "template" --nEpochs 50

	python train.py --cuda --gpu "0" --dataset "CAVE" --upscale_factor 4 --model_name "pde-net" --nEpochs 100 --resume checkpoints/CAVE_x4/template_4_epoch_50.pth

### Testing

	python test.py --cuda --gpu "0" --dataset "CAVE" --model_name "pde-net" --upscale_factor 4 --checkpoint checkpoints/CAVE_x4/pde-net_4_epoch_100.pth

Citation 
--------
**Please consider cite our work if you find it helpful.**

	@article{hou22deep,
		title={Deep Posterior Distribution-based Embedding for Hyperspectral Image Super-resolution},
		author={Hou, Jinhui and Zhu, Zhiyu and Hou, Junhui and Zeng, Huanqiang and Wu, Jinjian and Zhou, Jiantao},
		journal={IEEE Transactions on Image Processing},
		volume={31},
		number={},
		pages={5720-5732},
		year={2022},
		doi={10.1109/TIP.2022.3201478}
	}
  
