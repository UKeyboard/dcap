### Prepare the ImageNet Dataset

The [ImageNet](http://www.image-net.org/) project contains millions of images and thousands of objects for image classification. The dataset has multiple versions. And, the one most commonly used for image classification is [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/). 

> 300 GB disk space is required to download and extract ILSVRC 2012.


#### Step 1. Download dataset splits

- Register at [ImageNet](http://image-net.org/)
- Download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` like this:
```shell
aria2c -c -s 16 [URL]
```
- After downloading the file, use `md5sum [FILE]` to compare the hash with the provided hash. If it differs, download the file again.



#### Step 2. Extract the training split

The ILSVRC 2012 training split `ILSVRC2012_img_train.tar` contains 1000 files of the form *`n01440764.tar`*, *`n01443537.tar`*, etc. Each of the tar file contains JPEGs of one class.

Run the following script to extract `ILSVRC2012_img_train.tar`:

```shell
mkdir ISLVRC2012 && mv ILSVRC2012_img_train.tar ISLVRC2012/ && cd ISLVRC2012 && mkdir train
tar -xvf ILSVRC2012_img_train.tar -C train && rm -f ILSVRC2012_img_train.tar && cd train
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

The script will extract images w.r.t each class into the corresponding category folder, eg `n01440764`. It will delete the downloaded `ILSVRC2012_img_train.tar` as well. We recommend you to keep this tar file locally, and use the following script:

```shell
mkdir ISLVRC2012 && mv ILSVRC2012_img_train.tar ISLVRC2012/ && cd ISLVRC2012 && mkdir train
tar -xvf ILSVRC2012_img_train.tar -C train && cd train
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

#### Step 3. Extract the validation split
The validation split `ILSVRC2012_img_val.tar` contains 50,000 images (50 images per class), e.g. 
```txt
ILSVRC2012_val_00000001.JPEG
ILSVRC2012_val_00000002.JPEG
...
ILSVRC2012_val_00049999.JPEG
ILSVRC2012_val_00050000.JPEG
```

The ground truth of the validation images is in `data/ILSVRC2012_validation_ground_truth.txt` in ILSVRC 2012 develop kit [ILSVRC2012_devkit_t12.tar](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar), where each line contains one ILSVRC2012_ID for one image, in the ascending alphabetical order of the image file names. e.g.:
```txt
490
361
171
822
297
...
...
10
495
128
848
186
```
The *490* in the first line of `ILSVRC2012_validation_ground_truth.txt` indicates that the label of image `ILSVRC2012_val_00000001.JPEG` is 490 and the corresponding synset is in line 490 of `ILSVRC2012_mapping.txt`, i.e. *n01751748*. The synset word is *sea snake* in `mapping.txt`.

In practice, there exist other versions of label mapping, e.g. the caffe version [http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz](http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz). This tar contains files as follows:
```txt
det_synset_words.txt
imagenet.bet.pickle
imagenet_mean.binaryproto
synset_words.txt
synsets.txt
test.txt
train.txt
val.txt
```

The `val.txt` file is the label file of validation images. Here is an example of its content:
```txt
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
...
...
```
Each line contains a *\<name label\>* pair. 

Take the first line for instance, it says the label of image *ILSVRC2012_val_00000001.JPEG* is 65. We proceed to `synset.txt` and get the corresponding synset. **If 65 is zero-based, the synset is in line 66. Otherwise, it is in line 65 if the label is one-based**. The label in caffe version is zero-based, so the synset is in line 66, i.e. *n01751748*. Then we open `synset_words.txt`, search *n01751748* and get its synset word "sea snake".

**It's important to know the mapping used by a pretrained model.**

We can run the script ( The `ISLVRC2012` directory is created.)
```shell
mv ILSVRC2012_img_val.tar ISLVRC2012/ && cd ISLVRC2012 && mkdir val
tar -xvf ILSVRC2012_img_val.tar -C val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
or 
```shell
mv ILSVRC2012_img_val.tar ISLVRC2012/ && cd ISLVRC2012 && mkdir val
tar -xvf ILSVRC2012_img_val.tar -C val
wget -qO- https://files-cdn.cnblogs.com/files/luruiyuan/valprep.sh | bash
```
to put images belonging to the same class into the same subdirectory named by its synset.



### Acknowledgement
- [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-training.md)
- [GLUON AI: Prepare the ImageNet dataset](https://cv.gluon.ai/build/examples_datasets/imagenet.html)
- [How to download ImageNet](https://martin-thoma.com/download-data/)
- [fh295/semanticCNN ImageNet Labels](https://github.com/fh295/semanticCNN/tree/master/imagenet_labels)