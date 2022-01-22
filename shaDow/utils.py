# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import random
import string

import numpy as np
import yaml
from copy import deepcopy

import numbers
from typing import List
from graph_engine.frontend import TRAIN, VALID, TEST
from shaDow.globals import git_rev, timestamp


def parse_n_prepare(task, args, name_graph, dir_log, os_='linux'):
    # [config]
    if args.configs is not None:
        config_train = args.configs
    else:
        assert task in ['inference', 'postproc']
        if task == 'inference':
            if args.inference_configs is None:
                assert not args.compute_complexity_only
                dir_candy = args.inference_dir
            else:
                assert args.inference_dir is None and args.compute_complexity_only
                dir_candy = None
                config_train = args.inference_configs
        else: 
            if args.postproc_dir is not None:
                dir_candy = args.postproc_dir
            else:
                with open(args.postproc_configs) as f:
                    config_temp = yaml.load(f, Loader=yaml.FullLoader)
                if 'dir_pred_mat' in config_temp:   # all such dirs MUST contain the same yaml
                    dir_candy = config_temp['dir_pred_mat'][0]  
                elif 'dir_emb_mat' in config_temp:  # all ens models should have the same arch (only differs in sampler)
                    dir_candy = next(iter(config_temp['dir_emb_mat'].values()))[0]
                else:
                    raise NotImplementedError
        if dir_candy is not None:
            assert os.path.isdir(dir_candy)
            f_yml = [f for f in os.listdir(dir_candy) if f.split('.')[-1] in ['yml', 'yaml']]
            assert len(f_yml) == 1
            config_train = f"{dir_candy}/{f_yml[0]}"
    with open(config_train) as f_config_train:
        config_train = yaml.load(f_config_train, Loader=yaml.FullLoader)
    config_train_copy = deepcopy(config_train)
    # [data]
    config_data = {
        "to_undirected"  : False,
        "transductive"   : False,
        "norm_feat"      : True,
        "valedges_as_input": False
    }
    config_data.update(config_train['data'])
    # [arch]
    arch_gnn = {        # default values
        "dim"               : -1,
        "aggr"              : "sage",
        "residue"           : "none",
        "pooling"           : "center",
        "loss"              : "softmax",
        "num_layers"        : -1,
        "num_cls_layers"    : 1,            # 1 MLP layer for classifier on the node representation
        "act"               : "I",
        "layer_norm"        : 'norm_feat',
        "heads"             : -1,
        "feature_augment"   : "hops",
        "feature_augment_ops": 'sum',
        "feature_smoothen"  : "none",
        "label_smoothen"    : "none",        # label_smoothen is only considered if use_label != none
        "ensemble_act"      : "leakyrelu",
        "branch_sharing"    : False,
        "use_label"         : "none"
    }
    arch_gnn.update(config_train["architecture"])
    for k, v in arch_gnn.items():
        if type(v) == str:
            arch_gnn[k] = v.lower()
    assert arch_gnn['aggr'] in ['sage', 'gat', 'gatscat', 'gcn', 'mlp', 'gin', 'sgc', 'sign']
    assert arch_gnn['use_label'] in ['all', 'none', 'no_valid']
    assert arch_gnn['pooling'].split('-')[0] in ['mean', 'max', 'sum', 'center', 'sort']
    assert arch_gnn['residue'] in ['sum', 'concat', 'max', 'none']
    assert arch_gnn['feature_augment'] in ['hops', 'pprs', 'none', 'hops-pprs', 'drnls']
    assert arch_gnn['feature_augment_ops'] in ['concat', 'sum']
    assert arch_gnn['layer_norm'] in ['norm_feat', 'pairnorm']
    if arch_gnn["feature_augment"] and arch_gnn["feature_augment"].lower() != "none":
        arch_gnn["feature_augment"] = set(k for k in arch_gnn["feature_augment"].split("-"))
    else:
        arch_gnn['feature_augment'] = set()
    # [params]
    params_train = {
        "lr"                : 0.01,
        "dropedge"          : 0.0,
        "ensemble_dropout"  : "none",
        "term_window_size"  : 1,
        "term_window_aggr"  : 'center',
        "percent_per_epoch" : {'train': 1., 'valid': 1., 'test': 1.}
    }
    params_train.update(config_train["hyperparameter"])
    params_train["lr"] = float(params_train["lr"])
    for m in ['train', 'valid', 'test']:
        if m not in params_train['percent_per_epoch']:
            params_train['percent_per_epoch'][v] = 1.
    for m in ['train', 'valid', 'test']:
        assert 0 <= params_train['percent_per_epoch'][m] <= 1.
    # [sampler]
    sampler_preproc, sampler_train = [], []
    for s in config_train['sampler']:
        phase = s.pop('phase')
        if phase == 'preprocess':
            sampler_preproc.append(s)
        elif phase == 'train':
            sampler_train.append(s)
        else:
            raise NotImplementedError
    batch_size = config_train["hyperparameter"]["batch_size"]
    config_sampler_preproc = {"batch_size": batch_size, "configs": sampler_preproc}
    config_sampler_train = {"batch_size": batch_size, "configs": sampler_train}
    # add self-edges for certain arch. e.g., for GAT, will be divide-by-0 error in grad without self-edges
    if arch_gnn["aggr"] in ["gcn", "gat", "gatscat"]:
        for sc in config_sampler_train["configs"]:
            num_ens = [len(v) for k, v in sc.items() if k != 'method']
            assert max(num_ens) == min(num_ens)
            sc["add_self_edge"] = [True] * num_ens[0]
    # [copy yml]
    name_key = f"{arch_gnn['aggr']}_{arch_gnn['num_layers']}"
    if arch_gnn['num_cls_layers'] > 1:
        name_key += f"_{arch_gnn['num_cls_layers']}"
    dir_log_full = log_dir(task, config_train_copy, name_key, dir_log, name_graph, git_rev, timestamp)
    return params_train, config_sampler_preproc, config_sampler_train, config_data, arch_gnn, dir_log_full


def parse_n_prepare_postproc(dir_load, f_config, name_graph, dir_log, arch_gnn, logger):
    if f_config is not None:
        with open(f_config) as f:
            config_postproc = yaml.load(f, Loader=yaml.FullLoader)
        name_key = f"postproc-{arch_gnn['aggr']}_{arch_gnn['num_layers']}"
        log_dir('postproc', config_postproc, name_key, dir_log, name_graph, git_rev, timestamp)
    skip_instantiate = []
    if 'check_record' in config_postproc:
        load_acc_record = config_postproc['check_record']
    else:
        load_acc_record = True
    if config_postproc['method'] == 'cs':               # C&S
        acc_record = [] if load_acc_record else None
        if dir_load is not None:
            if 'dir_pred_mat' not in config_postproc:
                config_postproc['dir_pred_mat'] = [dir_load]
            elif os.path.realpath(dir_load) not in [os.path.realpath(pc) for pc in config_postproc['dir_pred_mat']]:
                config_postproc['dir_pred_mat'].append(dir_load)
        config_postproc['pred_mat'] = [None] * len(config_postproc['dir_pred_mat'])
        for i, di in enumerate(config_postproc['dir_pred_mat']):
            if load_acc_record:
                acc_record.append(logger.decode_csv('final', di))
            for f in os.listdir(di):
                if 'cs' == f.split('.')[-1] and f.startswith('pred_mat'):
                    config_postproc['pred_mat'][i] = torch.load(f"{di}/{f}")
                    break
        if all(m is not None for m in config_postproc['pred_mat']):
            skip_instantiate = ['data', 'model']
    elif config_postproc['method'] == 'ensemble':       # Variant of subgraph ensemble as postproc
        acc_record = {s: [] for s in config_postproc['dir_emb_mat']} if load_acc_record else None
        assert dir_load is None
        config_postproc['emb_mat'] = {k: [None] * len(v) for k, v in config_postproc['dir_emb_mat'].items()}
        for sname, dirs_l in config_postproc['dir_emb_mat'].items():
            for i, di in enumerate(dirs_l):
                if load_acc_record:
                    acc_record[sname].append(logger.decode_csv('final', di))
                for f in os.listdir(di):
                    if 'ens' == f.split('.')[-1] and f.startswith('emb_mat'):
                        config_postproc['emb_mat'][sname][i] = torch.load(f"{di}/{f}")
                        break
        if all(m is not None for s, mat_l in config_postproc['emb_mat'].items() for m in mat_l):
            skip_instantiate = ['model']        # you have to load data (role, labels) anyways
    return config_postproc, acc_record, skip_instantiate


def log_dir(task, config_new, yml_name_key, dir_log, name_graph, git_rev, timestamp):
    if task == 'train':
        prefix = 'running'
    elif task == 'inference':
        prefix = 'INF'
    elif task == 'postproc':
        prefix = 'POST'
    else:
        raise NotImplementedError
    _rand_tie_breaker = ''.join(random.sample(string.ascii_letters + string.digits, 4))
    log_dir = f"{dir_log}/{name_graph}/{prefix}/{timestamp}-{git_rev.strip():s}_RAND{_rand_tie_breaker}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    yml_file = f"{log_dir}/{yml_name_key}.yml"
    with open(yml_file, 'w') as f:
        yaml.dump(config_new, f, default_flow_style=False, sort_keys=False)
    return log_dir


def merge_stat_record(dict_l : List[dict]):
    key_l = [set(d.keys()) for d in dict_l]
    assert all(k == key_l[0] == set([TRAIN, VALID, TEST]) for k in key_l)
    names_stat = set(dict_l[0][TRAIN].keys())
    ret = {n: {TRAIN: [], VALID: [], TEST: []} for n in names_stat}
    for d in dict_l:
        for m in [TRAIN, VALID, TEST]:
            assert set(d[m].keys()) == names_stat
            for k, v in d[m].items():
                ret[k][m].append(v)
    return ret


# below are some functions used for general layer normalization

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def idx_nD_list(l, idx):
    """
    l is a multi-dimensional list. idx is a tuple indexing into l. 
    Since a python list cannot be indexed by a tuple directly,
    we use this function for indexing an arbitrary dim list.
    """
    if len(idx) == 0:
        return l
    idx = list(idx)
    i = idx.pop(0)
    return idx_nD_list(l[i], idx)

def set_nD_list(l, idx, val):
    if len(idx) == 1:
        l[idx[0]] = val
        return
    idx = list(idx)
    i = idx.pop(0)
    return set_nD_list(l[i], idx, val)

def construct_nD_list(shape, init_val=None):
    assert init_val is None or isinstance(init_val, numbers.Number)
    ret = [init_val]
    for d in shape[::-1]:
        if type(ret[0]) == list:
            ret = [ret[0].copy() for _ in range(d)]
        else:
            ret = [ret[0] for _ in range(d)]
        ret = [ret]
    return ret[0]
