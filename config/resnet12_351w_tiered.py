import torch
import torch.nn as nn

# set expriment env
is_apply_final_activation = True
is_maxpool_downsample = True
is_biased_classifier = True
is_dense = False
is_softlabel = False
is_augtrain_plus = True
is_multi_lr_decay = True
is_transductive_bn = False
drop_rate = 0.0
batch_size = 256
conv_out_dim = 512
max_num_epochs = 100

# build configure
config = {}
config["name"] = '_'.join([
    "resnet12_351w_tiered", 
    'augplus' if is_augtrain_plus else 'none',
    'multi' if is_multi_lr_decay else 'none',
    'max' if is_maxpool_downsample else 's2', 
    'dense' if is_dense else 'avg',
    'softlabel' if is_dense and is_softlabel else 'hardlabel',
    'wAC' if is_apply_final_activation else "woAC", 
    'wBias' if is_biased_classifier else 'woBias',
    'wTBN' if is_transductive_bn else 'woTBN',
    'wDrop' if drop_rate > 0 else 'woDrop',
    'b%d'%batch_size,
    'e%d'%max_num_epochs,
    ])
config["max_num_epochs"] = max_num_epochs
config["evaluate_step"] = 1 # evaluate the model every 5 epochs
config["best_metric"] = "Top1Acc" # select best model based on this metric

# solver
config["solver"] = dict(
    package = "algorithm.basic",
    module = ("DenseClassifierSoftlabelSolver" if is_softlabel else "DenseClassifierSolver") if is_dense else "ClassifierSolver",
    min_lr_decay = 5e-4,
)

# model
config['network'] = dict(
    package = "model",
    module = ("ResNet12_wDCplus" if is_softlabel else "ResNet12_wDC") if is_dense else "ResNet12_wC",
    param = dict(
        required_input_size = 84,
        num_classes=351, 
        biased=is_biased_classifier,
        maxpool=is_maxpool_downsample,
        drop_rate=max(0.0, drop_rate),
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False if is_dense else True,
        transductive_bn=is_transductive_bn,
        final_activation=nn.Sequential(
            nn.BatchNorm2d(conv_out_dim, track_running_stats=not is_transductive_bn),
            nn.LeakyReLU(0.1),
        ) if is_apply_final_activation else None
    ),
    pretrained = None, # or /path/to/pretrained/model
    freeze_pretrained = False, # only works when "pretrained" is set
)

# optimizer
config['optimizer'] = dict(
    package = "torch.optim",
    # module = "Adam",
    # param = dict(
    #     lr=0.001, 
    #     betas=(0.9, 0.999), 
    #     eps=1e-08, 
    #     weight_decay=5e-4, 
    #     amsgrad=False
    # ),
    module = "SGD",
    param = dict(
        lr=0.1, 
        momentum=0.9, 
        weight_decay=5e-4, 
        nesterov=True
    ),
)


# scheduler
config["scheduler"] = dict(
    package = "torch.optim.lr_scheduler",
    module = "MultiStepLR",
    param = dict(
        milestones = list(map(lambda x: int(x*max_num_epochs), [0.3, 0.6, 0.8] if is_multi_lr_decay else [0.8])),
        gamma = 0.1,
        last_epoch = -1
    ),
    # module = "ReduceLROnPlateau", # CURRENTLY DONOT SUPPORT
    # param = dict(
    #     mode='max',
    #     factor=0.1,
    #     patience=10,
    #     verbose=False,
    #     threshold=0.001,
    #     threshold_mode='rel',
    #     cooldown=0,
    #     min_lr=0,
    #     eps=1e-08
    # ),
)

# data
lmdb_file = '/cloud/hejun/dataset/imagenet/ILSVRC2012_img_train_x96.lmdb'
config["data"] = dict(
    package = "dataset",
    module = "TieredImagenetLMDBHorizontal",
    train = dict(
        phase = 'train', 
        transform_name = 'aug_train+' if is_augtrain_plus else 'aug_train',
        factor = 0.9,
        category_pool_name = "train",
        label_indent = 0,
        batch_size = batch_size,
        shuffle = True,
        lmdb = lmdb_file
    ),
    val = dict(
        phase = 'test', 
        transform_name = 'basic',
        factor = 0.9,
        category_pool_name = "train",
        label_indent = 0,
        batch_size = batch_size,
        lmdb = lmdb_file
    ),
    test = None
)