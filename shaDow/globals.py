# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import datetime, time
import subprocess
import numpy as np
import torch

# [STEP 0]
# We use the CONFIG.yml file to define the meta specification. 
# So parse the CONFIG.yml before we ever do anything. 
import yaml
from os import path
fname_global_config = 'CONFIG.yml'
if not path.exists(fname_global_config):
    fname_global_config = 'CONFIG_TEMPLATE.yml'
    assert path.exists(fname_global_config)
    print(f"LOADING {fname_global_config}. PLS DOUBLE-CHECK!!")
with open(fname_global_config) as f_config:
    meta_config = yaml.load(f_config, Loader=yaml.FullLoader)
DATA_METRIC = meta_config['data']['metric']
# for deterministic samplers such as PPR, we store the subgraphs in epoch 1 
# and then simply reuse those in later epochs
REUSABLE_SAMPLER = set(meta_config['algorithm']['sampler']['deterministic'])

args_logger = meta_config['logging']['logger']
cls_logger = args_logger.pop('name').split('.')
if len(cls_logger) == 1:
    exec(f"import shaDow.{cls_logger[0]} as Logger")
else:
    exec(f"from shaDow.{'.'.join(cls_logger[:-1])} import {cls_logger[-1]} as Logger")

# [STEP 1] parse args
parser = argparse.ArgumentParser(description="argument for ShallowSampling training / inference")
parser.add_argument("--dataset", required=True, choices=list(meta_config['data']['metric'].keys()), type=str, help="name of data")
parser.add_argument("--configs", required=False, type=str, default=None, help="path to the configuration of training (*.yml)")
parser.add_argument("--reload_model_dir", required=False, type=str, default=None, help="reload saved checkpoint")
parser.add_argument("--dtype", required=False, choices=['float32', 'float64', 'float16', 'bfloat16'], type=str, default='float32', help='[not yet supported] dtype for training')
parser.add_argument("--gpu", default=None, type=int, help="which GPU to use")
parser.add_argument("--eval_train_every", default=15, type=int, help="How often to evaluate training subgraph accuracy",)
parser.add_argument("--log_test_convergence", type=int, default=1, help="how often to show the test accuracy during training")
parser.add_argument("--timestamp", type=str, required=False, default=None)
parser.add_argument("--no_pbar", action='store_true', required=False, help="don't show tqdm progress bar")
parser.add_argument("--no_log", action="store_true", required=False, help="[for debug] set it to true when you don't want to log convergence")
parser.add_argument("--seed", type=int, default=-1, required=False, help="[for debug] fix seed for both python and C++ execution")
parser.add_argument("--set_cuda_deterministic", action="store_true", required=False, help="[for debug] set cuda to deterministic for fully reproducing prev run")
# for training speed-memory tradeoff
parser.add_argument("--nocache", type=str, default=None, choices=['train', 'valid', 'test', 'all'], help="don't cache the subgraph samples during phase train / val / test")
parser.add_argument("--full_tensor_on_gpu", action='store_true', required=False, help='if true, will leave the full graph node feat on gpu')
# inference only on a pre-trained model
parser.add_argument("--inference_dir", default=None, type=str, help="path to the model to reload. 'RANDOM' if you just want to use randomly initialized model without reload")
parser.add_argument("--inference_configs", default=None, type=str, help="only specify this if you ONLY want to compute inference complexity")
parser.add_argument("--inference_budget", default=None, type=int, help="how many root nodes to generate embeddings?")
parser.add_argument("--inference_log_meta", default="", type=str, help="extra info to be added to logs")
parser.add_argument("--is_inf_train", action="store_true", default=False, help="if we want to compute inf acc on training set")
parser.add_argument("--compute_complexity_only", action="store_true", default=False, help="if we only want to compute the cost of inference")
# post proc model: e.g., run C&S on a pre-trained model
parser.add_argument("--postproc_dir", default=None, type=str, help='path to pytorch checkpoint for post processing (e.g., C&S)')
parser.add_argument("--postproc_configs", default=None, type=str, help='yml to postprocessing algo')
Logger.add_logger_args(parser)

args_global = parser.parse_known_args()[0]

if args_global.seed >= 0:
    import random
    random.seed(args_global.seed)
    np.random.seed(args_global.seed)
    torch.manual_seed(args_global.seed)
    torch.cuda.manual_seed(args_global.seed)
    if args_global.set_cuda_deterministic:
        # torch.use_deterministic_algorithms(True) -- not supported for torch < 1.8
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

# [STEP 2] Special handling for Windows machine:
#       convert path from the `path\to\file` format to the `path/to/file` format
if meta_config['device']['software']['os'] == 'windows':
    for fn in ['configs', 'inference_dir', 'postproc_dir', 'postproc_configs']:
        path_temp = getattr(args_global, fn)
        if type(path_temp) == str:
            setattr(args_global, fn, path_temp.replace('\\', '/'))
fs_ignore = meta_config['logging']['ignore_config_name']
meta_config['logging']['ignore_config_name'] = [fs.replace('\\', '/') for fs in fs_ignore]

# [STEP 3] check if we do not want to log this run (e.g., we are in the development process
#       and the run is only meant for checking the program correctness and functionality. )
if not args_global.no_log and args_global.configs is not None:
    config_name = args_global.configs.split('/')[-1]
    import re
    for fs in meta_config['logging']['ignore_config_name']:
        if re.match(fs, config_name):
            args_global.no_log = True
            break

# [STEP 4] log git revision and timestamp to facilitate reproductivity. 
git_rev = subprocess.Popen(
    "git rev-parse --short HEAD",
    shell=True,
    stdout=subprocess.PIPE,
    universal_newlines=True,
).communicate()[0]
dtime_format = "%Y-%m-%d %H-%M-%S"
if not args_global.timestamp:
    timestamp = time.time()
    timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime(dtime_format)
else:
    timestamp = args_global.timestamp
args_global.timestamp = timestamp


# [STEP 5] auto choosing available NVIDIA GPU by querying nvidia-smi
gpu_selected = args_global.gpu
if gpu_selected is None:
    if 'gpu' in meta_config['device'] and meta_config['device']['gpu']['count'] > 0:
        if meta_config['device']['software']['os'] == 'linux':
            # select the one with most memory available
            # modify the `nvidia-smi` cmd if you want to select by a different criteria
            gpu_mem_raw = subprocess.Popen(
                "nvidia-smi -q -d Memory |grep -A4 GPU|grep Free",
                shell=True,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            ).communicate()[0].split("\n")
            memory_available = [int(x.split()[2]) for x in gpu_mem_raw if x]
            max_mem = np.argmax(memory_available)
            if memory_available[max_mem] < 200:
                gpu_selected = -1
            else:
                gpu_selected = max_mem
        elif meta_config['device']['software']['os'] == 'windows':
            gpu_selected = 0        # not yet supported auto GPU selection for Windows
        else:
            raise NotImplementedError
    else:
        gpu = -1
# if gpu_selected >= 0:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_selected)
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
args_global.gpu = gpu_selected
device = torch.device(f"cuda:{args_global.gpu}") if args_global.gpu >= 0 else torch.device("cpu")

print(f"SELECTED GPU {args_global.gpu}")

# handle data type - NOTE: dtype other than float32 has not been supported yet!!
if args_global.dtype == 'float32':
    torch.set_default_dtype(torch.float32)
elif args_global.dtype == 'float64':
    torch.set_default_dtype(torch.float64)
elif args_global.dtype == 'float16':
    torch.set_default_dtype(torch.float16)
elif args_global.dtype == 'bfloat16':
    torch.set_default_dtype(torch.bfloat16)
else:
    raise NotImplementedError 

