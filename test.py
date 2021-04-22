import os
import argparse
import numpy as np
import torch
from utils import import_module, import_x, set_random_seeds, repeatc

def options():
    parser = argparse.ArgumentParser(description="A general model evaluation script.")
    parser.add_argument('--config', type=str, required=True, default='',
        help='config parameters in /path/to/some/config.py')
    parser.add_argument('--checkpoint', type=int, default=0,
        help='checkpoint (epoch id) that will be loaded. Default load no checkpoint. '
            'If a negative value is given then the latest checkpoint is loaded.')
    parser.add_argument('--num_workers', type=int, default=4,
        help='number of data loading workers')
    parser.add_argument('--random_seed', type=int, default=2020,
        help='random seed')
    parser.add_argument("--gpu", action='append', default=[], help="gpu(s) to use, default []")
    parser.add_argument('--tqdm', action='store_true', default=False, help="show progress bar")
    parser.add_argument('--debug', action='store_true', default=False, help="running in debug mode")
    parser.add_argument('--noflog', action='store_true', default=False, help="if set, disable log to file.")
    parser.add_argument('--nosummary', action='store_true', default=False, help="if set, disable tensorboard summary.")
    parser.add_argument("--local_rank", type=int, help="Required for using the torch.distributed.launch utility.")
    args_opt = parser.parse_args()
    assert all(map(lambda x: int(x)>=0, args_opt.gpu))
    args_opt.iscuda = len(args_opt.gpu)>0
    if args_opt.iscuda:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=",".join(args_opt.gpu)
    args_opt.iscuda = args_opt.iscuda and torch.cuda.is_available()
    args_opt.ngpus = torch.cuda.device_count()
    return args_opt


if __name__ == "__main__":
    opts = options()
    is_multi_gpu = (opts.iscuda and opts.ngpus > 1)
    if is_multi_gpu:
        # prepare multi-GPU environment
        if not torch.distributed.is_available():
            raise RuntimeError("Requires distributed package to be available")
        torch.distributed.init_process_group(backend="nccl")
    set_random_seeds(opts.random_seed)
    # Load the configuration params of the experiment
    try:
        config = import_module(opts.config, "config").config
    except Exception as e:
        print('Loading experiment config:', opts.config, repeatc(".", 12), "failed.")
        raise e
    print('Loading experiment config:', opts.config, repeatc(".", 12), "done.")

    #
    config["exp_dir"] = os.path.join(".", "experiments", opts.config)
    print("Logs, snapshots and model files will be stored:", config["exp_dir"])
    Dataloader = import_x(config["data"]["module"], import_module(config["data"]["package"]))
    # dataloader_train = None if config["data"].get("train", None) is None else Dataloader(**(config["data"]["train"]), num_workers = opts.num_workers, distributed=is_multi_gpu)
    # dataloader_val = None if config["data"].get("val", None) is None else Dataloader(**(config["data"]["val"]), num_workers = opts.num_workers, distributed=is_multi_gpu)
    dataloader_test = None if config["data"].get("test", None) is None else Dataloader(**(config["data"]["test"]), num_workers = opts.num_workers, distributed=is_multi_gpu)
    #
    # update configure
    config.update(dict(
        checkpoint = '*' if opts.checkpoint < 0 else opts.checkpoint,
        iscuda = opts.iscuda,
        ngpus = opts.ngpus,
        tqdm = opts.tqdm,
        debug = opts.debug,
        noflog = opts.noflog,
        nosummary = opts.nosummary,
        local_rank = opts.local_rank
    ))
    T = import_x(config["solver"]["module"], import_module(config["solver"]["package"]))
    solver = T(config)
    # solver.solve(dataloader_train, dataloader_val, dataloader_test)
    solver.evaluate(dataloader_test)
