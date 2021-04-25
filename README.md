## A General Framework for Few-Shot Learning

This is an easy-to-use framework for few-shot learning. 

### Requirements 
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

### Supported Solvers
| Name | Module | Description |
| --- | ---- | ---- |
| GAP-pretraining | `algorithm.basic.ClassifierSolver`| A simple multi-category classifier. |
| DC-pretraining | `algorithm.basic.DenseClassifierSolver` | A simple multi-category classifier with dense classification. |
| Prototypical Networks | `algorithm.fewshot.ProtoNetxxxxSolver` | Please refer to ["Prototypical Networks for Few-shot Learning"](https://proceedings.neurips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf). |


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



### How to use the datasets?
For each dataset, we present two Dataloaders --- one for standard classification and the other for few-shot learning. 

For example, `MiniImagenetHorizontal` (for standard classification) and `MiniImagenetFS` (for few-shot learning) are two Dataloaders for MiniImagenet. 

For MiniImagenet, TieredImagenet and CUB-200-2011, we also present Dataloaders that load data from the datasets in LMDB format.

#### To use MiniImagenet dataset
For MiniImagenet, we present 4 Dataloaders --- `MiniImagenetFS`, `MiniImagenetLMDBFS`, `MiniImagenetHorizontal` and `MiniImagenetLMDBHorizontal`.

The first two dataloaders load images from ImageNet data folder, e.g. `ISLVRC2012/train`. And, the last two load data from the lmdb dataset, e.g. `/cloud/hejun/dataset/imagenet/ILSVRC2012_img_train.lmdb`.

Here are examples of using the four Dataloaders.

- `MiniImagenetFS`

```python
dataloader = MiniImagenetFS(
    version = 'protonet',
    phase = 'train', 
    transform_name = 'basic',
    label_indent = 0, 
    n_batch = 1000, # number of batches per epoch
    n_class = 5,
    n_support = 1,
    n_query = 15,
    n_unlabel = 0,
    batch_size = 4, # number of tasks per batch 
    imagenet = '/path/to/ISLVRC2012/train'
)
```

In the example, we create a dataloader that randomly samples 1,000 batches of 5-way 1-shot few-shot tasks from the meta-training split of MiniImagenet. Each batch contains 4 tasks. In other words, the dataloader sample 4,000 few-shot tasks per epoch. 

It is recommended to use `for` loop when working with large number of few-shot tasks, say 200k:
```python
for _ in range(50): # 50 epochs
    for i,batch in enumerate(dataloader): # 4k tasks per epoch
        images, fs_labels, labels = batch
        # now you are free to write your magic codes below
```
*Since the 4,000 tasks in each epoch are randomly sampled, we totally sample 200k different tasks in the above example.*

- `MiniImagenetLMDBFS`

`MiniImagenetLMDBFS` differs from `MiniImagenetFS` in that it is designed specifically for loading data from lmdb files, e.g.:
```python
dataloader = MiniImagenetLMDBFS(
    version = 'protonet',
    phase = 'val', 
    transform_name = 'basic',
    label_indent = -64,
    n_batch = 1000,
    n_class = 5,
    n_support = 1,
    n_query = 15,
    n_unlabel = 0,
    batch_size = 4,
    lmdb = '/path/to/imagenet/ILSVRC2012_img_train.lmdb'
)
```
In this example, we randomly 4k 5-way 1-shot tasks from the meta-validation split. Note the changes in `phase` and `label_indent` options.

- `MiniImagenetHorizontal`

`MiniImagenetHorizontal` is used in the standard classification pre-training stage.

```python
dataloaderTrain = MiniImagenetHorizontal(
    version = 'protonet',
    phase = 'train', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    imagenet = '/path/to/ISLVRC2012/train'
)

dataloaderTest = MiniImagenetHorizontal(
    version = 'protonet',
    phase = 'test', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    imagenet = '/path/to/ISLVRC2012/train'
)
```

In this example, we use the 64 meta-training classes for standard classification, where 90% images of each class are used for model training and the remaining 10% images are used for model evaluation.

- `MiniImagenetLMDBHorizontal`

```python
dataloaderTrain = MiniImagenetLMDBHorizontal(
    version = 'protonet',
    phase = 'train', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    lmdb = '/path/to/imagenet/ILSVRC2012_img_train.lmdb'
)

dataloaderTest = MiniImagenetLMDBHorizontal(
    version = 'protonet',
    phase = 'test', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    lmdb = '/path/to/imagenet/ILSVRC2012_img_train.lmdb'
)
```

In order to use `ILSVRC2012_img_train.lmdb`, you have to build it manually. To this end, please the [instructions](#prepare-mini-tiered) and prepare ImageNet first, then run the following script:
```shell
python scripts/prepare_imagenet_lmdb.py -f /path/to/imagenet/ISLVRC2012/train -o /path/to/imagenet/ISLVRC2012/ILSVRC2012_img_train.lmdb -S 200000000000
``` 

After building `ILSVRC2012_img_train.lmdb`, you can additionally run:
```shell
python scripts/downsample_lmdb_database.py -f /path/to/imagenet/ISLVRC2012/ILSVRC2012_img_train.lmdb -o /path/to/imagenet/ISLVRC2012/ILSVRC2012_img_train_x96.lmdb -s 96 -S 63000000000
```
to build a smaller lmdb database for easy transferability and fast reading. It will downsample every image in the input databse specified by `-f` option by resizing the short edge to the target size. Note the `-S` option must change properly to hold the resultant smaller database.

#### To use TieredImagenet dataset

For TieredImagenet, we present 4 Dataloaders --- `TieredImagenetFS`, `TieredImagenetLMDBFS`, `TieredImagenetHorizontal` and `TieredImagenetLMDBHorizontal`.

The usage of TieredImagenet is similar to that of MiniImagenet, but with minor difference.

- `TieredImagenetFS`

```python
dataloader = TieredImagenetFS(
    phase = 'train', 
    transform_name = 'basic',
    label_indent = 0, 
    n_batch = 1000, # number of batches per epoch
    n_class = 5,
    n_support = 1,
    n_query = 15,
    n_unlabel = 0,
    batch_size = 4, # number of tasks per batch 
    imagenet = '/path/to/ISLVRC2012/train'
)
```

- `TieredImagenetLMDBFS`

```python
dataloader = TieredImagenetLMDBFS(
    phase = 'train', 
    transform_name = 'basic',
    label_indent = 0,
    n_batch = 1000,
    n_class = 5,
    n_support = 1,
    n_query = 15,
    n_unlabel = 0,
    batch_size = 4,
    lmdb = '/path/to/imagenet/ILSVRC2012_img_train.lmdb'
)
```

- `TieredImagenetHorizontal`

```python
dataloader = TieredImagenetHorizontal{
    phase = 'train', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    imagenet = '/path/to/ISLVRC2012/train'
}
```

- `TieredImagenetLMDBHorizontal`

```python
dataloader = TieredImagenetLMDBHorizontal(
    phase = 'train', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    lmdb = '/path/to/imagenet/ILSVRC2012_img_train.lmdb'
)
```

#### To use CUB-200-2011 dataset

For CUB-200-2011, we also present 4 Dataloaders --- `CUB200FS`, `CUB200LMDBFS`, `CUB200Horizontal` and `CUB200LMDBHorizontal`.

- `CUB200FS`

```python
dataloader = CUB200FS(
    phase = 'train', 
    transform_name = 'basic',
    label_indent = 0, 
    n_batch = 1000, # number of batches per epoch
    n_class = 5,
    n_support = 1,
    n_query = 15,
    n_unlabel = 0,
    batch_size = 4, # number of tasks per batch 
    cub200dir = '/path/to/CUB-200-2011'
)
```

- `CUB200LMDBFS`

```python
dataloader = CUB200LMDBFS(
    phase = 'train', 
    transform_name = 'basic',
    label_indent = 0,
    n_batch = 1000,
    n_class = 5,
    n_support = 1,
    n_query = 15,
    n_unlabel = 0,
    batch_size = 4,
    lmdb = '/path/to/CUB-200-2011.lmdb'
)
```

- `CUB200Horizontal`

```python
dataloader = CUB200Horizontal{
    phase = 'train', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    cub200dir = '/path/to/CUB-200-2011'
}
```

- `CUB200LMDBHorizontal`

```python
dataloader = CUB200LMDBHorizontal(
    phase = 'train', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    lmdb = '/path/to/CUB-200-2011.lmdb'
)
```

For CUB-200-2011, similarly you have to build the lmdb databse manually if you're planing to use it.
```shell
python scripts/prepare_cub200_lmdb.py -f /path/to/CUB_200_2011.tgz -o /path/to/CUB_200_2011_woSeg.lmdb -j 1 -S 2073741824
```

Note the `_woSeg` suffix in the name of the output file. The suffix indicates that we donot use segmentation annotations, i.e. segmentation annotations are not availiable in the lmdb database. For more information, we refer the reader to `scripts/prepare_cub200_lmdb.py`.


#### To use CIFAR-FS and FC100 dataset

We present two Dataloaders for each dataset --- `CIFARFS100` and `CIFARFS100Horizontal` for CIFAR-FS, `CIFARFC100` and `CIFARFC100Horizontal` for FC100.

 - `CIFARFS100`

```python
dataloader = CIFARFS100(
    phase = 'train', 
    transform_name = 'basic',
    label_indent = 0,
    n_batch = 1000,
    n_class = 5,
    n_support = 1,
    n_query = 15,
    n_unlabel = 0,
    batch_size = 4,
    cifardir = '/path/to/cifar100'
 )
```

- `CIFARFS100Horizontal`

```python
dataloader = CIFARFS100Horizontal(
    phase = 'train', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    cifardir = '/path/to/cifar100'
)
```

- `CIFARFC100`

```python
dataloader = CIFARFC100(
    phase = 'train', 
    transform_name = 'basic',
    label_indent = 0,
    n_batch = 1000,
    n_class = 5,
    n_support = 1,
    n_query = 15,
    n_unlabel = 0,
    batch_size = 4,
    cifardir = '/path/to/cifar100'
 )
```

- `CIFARFC100Horizontal`

```python
dataloader = CIFARFC100Horizontal(
    phase = 'train', 
    transform_name =  'basic',
    factor = 0.9,
    category_pool_name = "train",
    label_indent = 0,
    batch_size = 256,
    shuffle = True,
    cifardir = '/path/to/cifar100'
)
```

In all the examples, we mainly illustrate how to obtain the meta-training data for few-shot learning or the training data for standard classification. We assume the reader can quickly figure out how to extend the usage to other situations like meta-validation, meta-testing, etc.


### Training

It takes at least two steps to kick off a training. The first step is to write a experiment configure file. And, the second step is to start training with configures defined in the configure file.

#### Prepare a configure file

Create a new configure file in directory `config` in which we specify the solver, network architecture, optimizer, learning rate scheduler and datasets for training, validation and testing.

Refer to `config/basic_protonet_plus_5w1s_tiered.py` for an instance.


#### Kick off the training

Run the script to kick off training:
```shell
python trainval.py --config basic_protonet_plus_5w1s_tiered --gpu 0 
```

The script `trainval.py` will try to find `basic_protonet_plus_5w1s_tiered.py` in directory `config` and load configures in file to set the experiment context. 

The option `--gpu 0` indicates to use the GPU card indexed by number 0. 

We use [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) for multi-GPU training. Here is an example of using three GPUs on the same server for training.

```shell
python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0 --config basic_protonet_plus_5w1s_tiered --gpu 0 1 3
```

If the batch size is set to 4 in the configure file, the true batch size is 4 x 3 when using three GPUs.


### Evaluation
By default, the model is evaluated on the validation set during training. Set `config['evaluate_step'] = 5` in the configure file to evaluate the model every 5 epochs.

**Note that the validation dataset is required.**


### Testing

If the testing dataset is set in the configure file, the model is also evaluated on the testing set after each evaluation on the validation set.

For example:

```python
# data
lmdb_file = '/cloud/hejun/dataset/imagenet/ILSVRC2012_img_train.lmdb'
config["data"] = dict(
    package = "dataset",
    module = "TieredImagenetLMDBFS",
    train = dict(
        phase = 'train', 
        transform_name = 'aug_train+' if is_augtrain_plus else 'aug_train',
        label_indent = 0,
        n_batch = int(1000/batch_size),
        n_class = config["episode_train_meta"].n_class,
        n_support = config["episode_train_meta"].n_support,
        n_query = config["episode_train_meta"].n_query,
        n_unlabel = config["episode_train_meta"].n_unlabel,
        batch_size = batch_size,
        lmdb = lmdb_file
    ),
    val = dict(
        phase = 'val', 
        transform_name = 'basic',
        label_indent = -351,
        n_batch = int(1000/batch_size),
        n_class = config["episode_test_meta"].n_class,
        n_support = config["episode_test_meta"].n_support,
        n_query = config["episode_test_meta"].n_query,
        n_unlabel = config["episode_test_meta"].n_unlabel,
        batch_size = batch_size,
        lmdb = lmdb_file
    ),
    # test = None,
    test = dict(
        phase = 'test', 
        transform_name = 'basic',
        label_indent = -448,
        n_batch = int(1000/batch_size),
        n_class = config["episode_test_meta"].n_class,
        n_support = config["episode_test_meta"].n_support,
        n_query = config["episode_test_meta"].n_query,
        n_unlabel = config["episode_test_meta"].n_unlabel,
        batch_size = batch_size,
        lmdb = lmdb_file
    )
)
```

Set `test = None` to disable model evaluation on the testing set during training in which case you have to manually start a testing session after training.

For manually testing, set the testing dataset first and then run
```python
python test.py --config basic_protonet_plus_5w1s_tiered --gpu 0 --checkpoint [CHECKPOINT]
```

### Others
By default, both training and testing will write summaries in `experiment/[NAME]/visuals` and logs in `experiment/[NAME]/logs/LOG_INFO_[TIME].txt`.

You can run `tensorboard --logdir experiment/[NAME]/visuals --port [PORT]` to visualize summaries.

To disable summaries, set `--nosummary` option. 

You can also set `--noflog` to print logs in the terminal only.

For better experience, we recommend to set the `--tqdm` option which will display progress bar during training and testing.