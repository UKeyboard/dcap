## Revisiting Deep Local Descriptor for Improved Few-Shot Classification
This repository contains the code to reproduce the few-shot classification experiments carried out in
[Revisiting Deep Local Descriptor for Improved Few-Shot Classification](https://arxiv.org/abs/2103.16009).

<br>

### Dependencies 
- python >= 3.7
- pytorch >= 1.1.0
- torchvision >= 0.3.0
- numpy >= 1.5
- pillow
- python-lmdb
- itertools 
- tqdm
- pickle
- matplotlib

<br>

### Hardware Requirements
-  linux server with at least two NVIDIA 1080Ti GPU cards
- 16GB RAM
- 320GB storage (for holding ILSVRC-12 dataset)


<br>

### Supported Datasets
| Dataset | Description |
| --- | ---- |
| MiniImagenet | The miniImagenet dataset is a derivative of the [ILSVRC-12 or ImageNet-1K](http://image-net.org/download-images) dataset. It consists of 60,000 color images of size 84 x 84 that are divided into 100 classes with 600 images each. The dataset is split into 64 base classes for meta-training, 16 novel classes for meta- validation, and 20 novel classes for meta-testing.|
| TieredImagenet | The tieredImagenet dataset is another derivative of the ILSVRC-12 or ImageNet-1K dataset. It contains more than 700,000 images, divided into 608 classes with the average number of images in each class being more than 1200. It has a hierarchical structure with all its 608 classes derived from 34 high-level nodes in ILSVRC-12. The 34 top categories are divided into 20 meta-training (351 classes), 6 meta-validation (97 classes) and 8 meta-testing (160 classes) categories. This high-level split provides a more challenging and realistic few- shot setting where the meta-training set is distinctive enough from the meta-testing set semantically.|
| CUB-200-2011 | The CUB-200-2011 dataset contains all 200 classes from [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). All the 11788 color images are divided into 100 classes for meta-training, 50 classes for meta-validation and 50 classes for meta-testing, respectively. |
| CIFAR-FS | The CIFAR-FS dataset is a derivative of the original [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset by randomly splitting 100 classes into 64, 16 and 20 classes for meta-training, meta-validation, and meta-testing with each class contains 600 images. |
| FC100 | The FC100 dataset also contains all 100 classes from CIFAR-100 where the classes are first are grouped into 20 high-level classes and then further divided into 60 classes from 12 high-level classes for meta-training, 20 classes from 4 high-level classes for meta-validation and 20 classes from 4 high-level classes for meta-testing, in a similar high-level split way to tieredImagNet. |

<br>

### Prepare Datasets

####  <span id='prepare-mini-tiered'>MiniImagenet and TieredImagenet</span>
MiniImagenet and TieredImagenet are two derivatives of ILSVRC12 dataset. To build the two datasets, the first step is to click [this link](http://image-net.org/download-images) and download the ILSVRC12 dataset, specifically download the train split `ILSVRC2012_img_train.tar`.

Run the following script to extract `ILSVRC2012_img_train.tar`. 
```shell
mkdir ISLVRC2012 && mkdir ISLVRC2012/train
tar -xvf ILSVRC2012_img_train.tar -C ISLVRC2012/train && cd ISLVRC2012/train
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

####  CUB-200-2011
For CUB-200-2011, download [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) ([click to download](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)) and run
```shell
mkdir CUB-200-2011
tar -xzvf CUB_200_2011.tgz -C CUB-200-2011 && cd CUB-200-2011
```

Note that we donot use the [segmantation annotation](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz).

####  CIFAR-FS and FC100
The CIFAR-FS and FC100 are based on [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html). 

For CIFAR100, we use `torchvision.datasets.cifar.CIFAR100`. The module will download CIFAR100 dataset for us.

<br>

### Usuage

To train and test DCAP on the TieredImagenet dataset:

1. First edit `config/resnet12_351w_tiered.py` and enable dense classification.
2. Run `python -m torch.distributed.launch --nproc_per_node=2  --nnodes=1 --node_rank=0 trainval.py --config resnet12_351w_tiered --gpu 0 1` to kick off the pre-training.
3. Edit `config/resnet12_dcap_plus_5w1s_tiered.py` and `config/resnet12_dcap_plus_5w5s_tiered.py` to set pre-trained model properly.
4. For *5-way 1-shot* meta-training, meta-evaluation and meta-testing: `python --config resnet12_dcap_plus_5w1s_tiered --gpu 0 --tqdm`.
5. For *5-way 1-shot* meta-training, meta-evaluation and meta-testing: `python --config resnet12_dcap_plus_5w5s_tiered --gpu 0 --tqdm`.

<br>

### Others
By default, both training and testing will write summaries in `experiment/[NAME]/visuals` and logs in `experiment/[NAME]/logs/LOG_INFO_[TIME].txt`.

You can run `tensorboard --logdir experiment/[NAME]/visuals --port [PORT]` to visualize summaries in a browser.

To disable summaries, set the `--nosummary` option. 

You can also set the `--noflog` option to print logs in the terminal only.

<br>

### Contact
To ask questions or report issues, please open an issue on the issues tracker.

<br>

### Citation
If you use this code, please cite our [DCAP](https://arxiv.org/abs/2103.16009) paper:

```
@article{he2021revisiting,
  title={Revisiting Deep Local Descriptor for Improved Few-Shot Classification},
  author={He, Jun and Hong, Richang and Liu, Xueliang and Xu, Mingliang and Wang, Meng},
  journal={arXiv preprint arXiv:2103.16009},
  year={2021}
}
```