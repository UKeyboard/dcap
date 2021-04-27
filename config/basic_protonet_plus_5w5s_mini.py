import torch
import torch.nn as nn
from dataset import FewShotEpisodeMeta

# set expriment env
is_apply_final_activation = True
is_biased_classifier = True
is_meta_fintuning = True
is_dense_pretrained = False
is_dense_pretrained_w_softlabel = False and is_dense_pretrained
is_dense = False # If true, use dense classification as auxilary task
is_augtrain_plus = False
is_transductive_bn = False
is_cosine_solver = False
batch_size = 4
conv_out_dim = 64
pretrained_models = [
[
# dense pre-trained models
None, # dense pretrained without final activation
'experiments/basic_64w_mini/basic_64w_mini_conv64_none_none_dense_hardlabel_wAC_wBias_woTBN_b256_e100_ckpt_epoch89.best', # dense pretrained with final activation
],[
# gap pre-trained models
None, # gap pretrained without final activation
'experiments/basic_64w_mini/basic_64w_mini_conv64_none_none_avg_hardlabel_wAC_wBias_woTBN_b256_e100_ckpt_epoch87.best', # gap pretrained with final activation
]
]
if is_dense_pretrained:
    pretrained_model = pretrained_models[0][1] if is_apply_final_activation else pretrained_models[0][0]
else:
    pretrained_model = pretrained_models[1][1] if is_apply_final_activation else pretrained_models[1][0]
pretrained_model = pretrained_model if is_meta_fintuning else None
#
experiment_uuid = '_'.join([
    'dense' if is_dense_pretrained else 'avg',
    'softlabel' if is_dense_pretrained_w_softlabel else 'hardlabel',
    'wAC' if is_apply_final_activation else 'woAC',
    'wBias' if is_biased_classifier else 'woBias',
    'wTBN' if is_transductive_bn else 'woTBN',
])
if pretrained_model is not None:
    # make sure model has the target structure and is pre-trained as stated.
    assert experiment_uuid in pretrained_model
    assert 'basic_64w_mini_conv%d'%conv_out_dim in pretrained_model
else:
    is_meta_fintuning = False
    experiment_uuid = '_'.join([
        'none',
        'hardlabel',
        'wAC' if is_apply_final_activation else 'woAC',
        'wBias' if is_biased_classifier else 'woBias',
        'wTBN' if is_transductive_bn else 'woTBN',
    ])
#
max_num_epochs = 20 if is_meta_fintuning else 100


# build configure
config = {}
config["name"] = '_'.join([
    "basic_protonet_plus_5w5s_mini", 
    'conv%d'%conv_out_dim,
    'augplus' if is_augtrain_plus else 'none',
    #
    # 'dense' if is_dense_pretrained else 'avg',
    # 'softlabel' if is_dense_pretrained_w_softlabel else 'hardlabel',
    # 'wAC' if is_apply_final_activation else "woAC", 
    # 'wBias' if is_biased_classifier else 'woBias',
    # 'wTBN' if is_transductive_bn else 'woTBN',
    #
    experiment_uuid,
    'wDC' if is_dense else 'woDC',
    'wFT' if is_meta_fintuning else 'woFT',
    'cosine' if is_cosine_solver else 'project',
    'b%d'%batch_size,
    'e%d'%max_num_epochs,
    ])
config["max_num_epochs"] = max_num_epochs
config["evaluate_step"] = 1 # evaluate the model every 5 epochs
config["best_metric"] = "Top1Acc" # select best model based on this metric
config["episode_train_meta"] = FewShotEpisodeMeta(5,5,10 if batch_size>2 else 15,0) # train episode setting
config["episode_test_meta"] = FewShotEpisodeMeta(5,5,10 if batch_size>2 else 15,0) # test episode setting

# solver
config["solver"] = dict(
    package = "algorithm.fewshot",
    module = ("ProtoNet%sMetricPlusPlusSolver" if is_dense else "ProtoNet%sMetricPlusSolver") % ("Cosine" if is_cosine_solver else "Projection"),
    min_lr_decay = 1e-4,
    vanilla_classifier_balance_weight = 0.5,
    fewshot_classifier_balance_weight = 1.0,
)

# model
config['network'] = dict(
    package = "model",
    module = ("ConvNet%d_wDCplus" if is_dense else "ConvNet%d_wCplus")%conv_out_dim,
    param = dict(
        required_input_size = 84,
        num_classes=64, 
        biased=is_biased_classifier,
        avgpool = not is_dense,
        transductive_bn=is_transductive_bn,
        final_activation=nn.Sequential(
            nn.BatchNorm2d(conv_out_dim, track_running_stats=not is_transductive_bn),
            nn.LeakyReLU(0.1),
        ) if is_apply_final_activation else None
    ),
    pretrained = pretrained_model, # or /path/to/pretrained/model
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
        lr=0.001 if is_meta_fintuning else 0.1, 
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
        milestones = list(map(lambda x: int(x*max_num_epochs), [0.8] if is_meta_fintuning else [0.2, 0.6, 0.8, 0.9])),
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
lmdb_file = '/cloud/hejun/dataset/miniImagenet/FSL_miniImagenet_x84.lmdb'
config["data"] = dict(
    package = "dataset",
    module = "MiniImagenetLMDBFS",
    train = dict(
        version = 'protonet',
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
        version = 'protonet',
        phase = 'val', 
        transform_name = 'basic',
        label_indent = -64,
        n_batch = int(1000/batch_size),
        n_class = config["episode_test_meta"].n_class,
        n_support = config["episode_test_meta"].n_support,
        n_query = config["episode_test_meta"].n_query,
        n_unlabel = config["episode_test_meta"].n_unlabel,
        batch_size = batch_size,
        lmdb = lmdb_file
    ),
    test = dict(
        version = 'protonet',
        phase = 'test', 
        transform_name = 'basic',
        label_indent = -80,
        n_batch = int(1000/batch_size),
        n_class = config["episode_test_meta"].n_class,
        n_support = config["episode_test_meta"].n_support,
        n_query = config["episode_test_meta"].n_query,
        n_unlabel = config["episode_test_meta"].n_unlabel,
        batch_size = batch_size,
        lmdb = lmdb_file
    )
)