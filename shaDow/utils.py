# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import glob

import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.preprocessing import StandardScaler

from shaDow.globals import git_rev, timestamp, Logger
from torch_scatter import scatter

from copy import deepcopy

from typing import List, Union
from shaDow import TRAIN, VALID, TEST

from shaDow.data_converter import convert2shaDow, to_undirected



def load_data(prefix, dataset, config_data, os_='linux'):
    Logger.printf("Loading training data..")
    prefix_l = prefix['local']
    fs_shadow = ['adj_full_raw.np[yz]', 'adj_train_raw.np[yz]', 'label_full.npy', 'feat_full.npy', 'split.npy']
    if not all(glob.glob(f"{prefix_l}/{dataset}/{f}") for f in fs_shadow):
        convert2shaDow(dataset, prefix_l)
    role = np.load(f"./{prefix_l}/{dataset}/split.npy", allow_pickle=True)
    if type(role) == np.ndarray:
        role = role[()]
    else:
        assert type(role) == dict
    # role is used as index, which is required to be int64 (node_set won't take much mem anyways)
    node_set = {TRAIN: np.asarray(role[TRAIN], dtype=np.int64), 
                VALID: np.asarray(role[VALID], dtype=np.int64), 
                TEST : np.asarray(role[TEST], dtype=np.int64)}
    # load adj. If we want to convert to_undirected, and the undirected adj has been stored as external file,
    # then we skip the conversion in the program and directly load the undirected adj. 
    bin_adj_files = {TRAIN: {'indptr': None, 'indices': None, 'data': None},
                     VALID: {'indptr': None, 'indices': None, 'data': None},
                     TEST:  {'indptr': None, 'indices': None, 'data': None}}
    def fill_bin_adj_dict(mode_, split_, type_):
        for d in ['indptr', 'indices', 'data']:
            bin_adj_files[mode_][d] = f"{prefix_l}/{dataset}/cpp/adj_{split_}_{type_}_{d}.bin"
    if config_data['to_undirected']:
        if (adj_full == load_adj(prefix_l, dataset, 'undirected', 'full')) is None:
            adj_full = load_adj(prefix_l, dataset, 'raw', 'full')
            adj_full = to_undirected(adj_full)
        fill_bin_adj_dict(VALID, 'full', 'undirected')
        fill_bin_adj_dict(TEST, 'full', 'undirected')
        if config_data['transductive']:
            adj_train = adj_full
            fill_bin_adj_dict(TRAIN, 'full', 'undirected')
        elif (adj_train == load_adj(prefix_l, dataset, 'undirected', 'train')) is None:
            adj_train = load_adj(prefix_l, dataset, 'raw', 'train')
            adj_train = to_undirected(adj_train)
            fill_bin_adj_dict(TRAIN, 'train', 'undirected')
            assert set(adj_train.nonzero()[0]).issubset(set(node_set[TRAIN]))
    else:
        adj_full = load_adj(prefix_l, dataset, 'raw', 'full')
        fill_bin_adj_dict(VALID, 'full', 'raw')
        fill_bin_adj_dict(TEST, 'full', 'raw')
        if config_data['transductive']:
            adj_train = adj_full
            fill_bin_adj_dict(TRAIN, 'full', 'raw')
        else:
            adj_train = load_adj(prefix_l, dataset, 'raw', 'train')
            assert set(adj_train.nonzero()[0]).issubset(set(node_set[TRAIN]))
            fill_bin_adj_dict(TRAIN, 'train', 'raw')

    bin_adj_files = validate_bin_file(bin_adj_files)

    Logger.printf(f"SETTING TO {'TRANS' if config_data['transductive'] else 'IN'}DUCTIVE LEARNING", style="red")
    label_full = np.load(f"./{prefix_l}/{dataset}/label_full.npy")
    label_full = torch.from_numpy(label_full)
    
    # ======= deal with feats =======
    mode_norm = 'all' if config_data['transductive'] else 'train'
    if config_data['norm_feat'] and os.path.isfile(f"./{prefix_l}/{dataset}/feat_full_norm_{mode_norm}.npy"):
        feats = np.load(f"./{prefix_l}/{dataset}/feat_full_norm_{mode_norm}.npy")
        Logger.printf(f"Loading '{mode_norm}'-normalized features", style='yellow')
    else:
        feats = np.load(f"./{prefix_l}/{dataset}/feat_full.npy")
        if config_data['norm_feat']:
            feats_fit = feats if config_data['transductive'] else feats[node_set[TRAIN]]
            scaler = StandardScaler()
            scaler.fit(feats_fit)
            feats = scaler.transform(feats)
            Logger.printf(f"Normalizing node features (mode = {mode_norm})", style="yellow")
        else:
            Logger.printf("Not normalizing node features", style="yellow")
    feats = torch.from_numpy(feats.astype(np.float32, copy=False))
    Logger.printf("Done loading training data..")
    return {'adj_full'  : adj_full, 
            'adj_train' : adj_train, 
            'feat_full' : feats, 
            'label_full': label_full, 
            'node_set'  : node_set,
            'bin_adj_files': bin_adj_files}


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
    config_data = {"to_undirected"  : False,
                   "transductive"   : False,
                   "norm_feat"      : True}
    config_data.update(config_train['data'])
    # [arch]
    arch_gnn = {        # default values
        "dim"               : -1,
        "aggr"              : "sage",
        "residue"           : "none",
        "pooling"           : "center",
        "loss"              : "softmax",
        "num_layers"        : -1,
        "act"               : "I",
        "heads"             : -1,
        "feature_augment"   : "hops",
        "feature_smoothen"  : "none",
        "label_smoothen"    : "none",        # label_smoothen is only considered if use_label != none
        "ensemble_act"      : "leakyrelu",
        "branch_sharing"    : False,
        "use_label"         : "none"
    }
    arch_gnn.update(config_train["architecture"])
    assert arch_gnn['aggr'] in ['sage', 'gat', 'gatscat', 'gcn', 'mlp', 'gin', 'sgc', 'sign']
    assert arch_gnn['use_label'].lower() in ['all', 'none', 'no_valid']
    assert arch_gnn['pooling'].lower().split('-')[0] in ['mean', 'max', 'sum', 'center', 'sort']
    assert arch_gnn['residue'].lower() in ['sum', 'concat', 'max', 'none']
    assert arch_gnn['feature_augment'].lower() in ['hops', 'ppr', 'none']
    if arch_gnn["feature_augment"] and arch_gnn["feature_augment"].lower() != "none":
        arch_gnn["feature_augment"] = set(k for k in arch_gnn["feature_augment"].split("-"))
    else:
        arch_gnn['feature_augment'] = set()
    # [params]
    params_train = {
        "lr"                : 0.01,
        "dropedge"          : 0.0,
        "ensemble_dropout"  : "none"
    }
    params_train.update(config_train["hyperparameter"])
    params_train["lr"] = float(params_train["lr"])
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
    log_dir = f"{dir_log}/{name_graph}/{prefix}/{timestamp}-{git_rev.strip():s}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    yml_file = f"{log_dir}/{yml_name_key}.yml"
    with open(yml_file, 'w') as f:
        yaml.dump(config_new, f, default_flow_style=False, sort_keys=False)
    return log_dir


# =============== #
#    ADJ UTILS    #
# =============== #

def get_deg_torch_sparse(adj):
    return scatter(adj._values(), adj._indices()[0], reduce="sum")


def adj_norm_rw(adj, deg=None, dropedge=0., sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    
    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    """
    if type(adj) == torch.Tensor:
        assert deg is None
        assert torch.sum(adj._values()).cpu().long().item() == adj._values().size()[0]
        _deg_orig = get_deg_torch_sparse(adj)
        if dropedge > 0:
            masked_indices = torch.floor(torch.rand(int(adj._values().size()[0] * dropedge)) * adj._values().size()[0]).long()
            adj._values()[masked_indices] = 0
            _deg_dropped = get_deg_torch_sparse(adj)
        else:
            _deg_dropped = _deg_orig
        _deg = torch.repeat_interleave(_deg_dropped, _deg_orig.long())
        _deg = torch.clamp(_deg, min=1)
        _val = adj._values()
        _val /= _deg
        adj_norm = adj
    else:
        assert dropedge == 0., "not supporting dropedge for scipy csr matrices"
        assert adj.shape[0] == adj.shape[1]
        diag_shape = (adj.shape[0], adj.shape[1])
        D = adj.sum(1).flatten() if deg is None else deg
        D = np.clip(D, 1, None)     # if deg_v == 0, it doesn't matter what value we clip it to. 
        norm_diag = sp.dia_matrix((1 / D, 0), shape=diag_shape)
        adj_norm = norm_diag.dot(adj)
        if sort_indices:
            adj_norm.sort_indices()
    return adj_norm


def adj_norm_sym(adj, sort_indices=True, add_self_edge=False, dropedge=0.):
    assert adj.shape[0] == adj.shape[1]
    assert adj.data.sum() == adj.size, "symmetric normalization only supports binary input adj"
    N = adj.shape[0]
    # drop edges symmetrically
    if dropedge > 0:
        masked_indices = np.random.choice(adj.size, int(adj.size * dropedge))
        adj.data[masked_indices] = 0
        adjT = adj.tocsc()
        data_add = adj.data + adjT.data
        survived_indices = np.where(data_add == 2)[0]
        adj.data *= 0
        adj.data[survived_indices] = 1
    # augment adj with self-connection
    if add_self_edge:
        indptr_new = np.zeros(N + 1)
        neigh_list = [set(adj.indices[adj.indptr[v] : adj.indptr[v+1]]) for v in range(N)]
        for i in range(len(neigh_list)):
            neigh_list[i].add(i)
            neigh_list[i] = np.sort(np.fromiter(neigh_list[i], int, len(neigh_list[i])))
            indptr_new[i + 1] = neigh_list[i].size
        indptr_new = indptr_new.cumsum()
        indices_new = np.concatenate(neigh_list)
        data_new = np.broadcast_to(np.ones(1), indices_new.size)
        adj_aug = sp.csr_matrix((data_new, indices_new, indptr_new), shape=adj.shape)
        # NOTE: no need to explicitly convert dtype, since adj_norm_sym is used for subg only
    else:
        adj_aug = adj
    # normalize
    D = np.clip(adj_aug.sum(1).flatten(), 1, None)
    norm_diag = sp.dia_matrix((np.power(D, -0.5), 0), shape=adj_aug.shape)
    adj_norm = norm_diag.dot(adj_aug).dot(norm_diag)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


def coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))


# ================= #
# ADJ FILE IO UTILS #
# ================= #

def load_adj(prefix, dataset, type_, split_):
    """
    Try to load the prestored undirected adj. If the file does not exist, then you MUST return a None
    """
    assert split_ in ['full', 'train'], "UNKNOWN ADJ SPLIT. ONLY ACCEPT [full] or [train]"
    assert type_ in ['raw', 'undirected'], "UNKNOWN ADJ TYPE. ONLY ACCEPT [raw] or [undirected]"
    file_adj = f"{prefix}/{dataset}/adj_{split_}_{type_}." + "{}"
    if os.path.isfile(file_adj.format('npz')):
        adj = sp.load_npz(file_adj.format('npz'))
    elif os.path.isfile(file_adj.format('npy')):
        adj_d = np.load(file_adj.format('npy'), allow_pickle=True)
        if type(adj_d) == np.ndarray:
            adj_d = adj_d[()]
        else:
            assert type(adj_d) == dict
        indptr = adj_d['indptr']
        indices = adj_d['indices']
        if 'data' in adj_d:
            data = adj_d['data']
        else:
            data = np.broadcast_to(np.ones(1, dtype=np.bool), indices.size)
        num_nodes = indptr.size - 1
        adj = sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
    else:
        adj = None
    return adj


def validate_bin_file(bin_adj_files):
    for md, df in bin_adj_files.items():
        assert set(df.keys()) == set(['indptr', 'indices', 'data'])
        if not os.path.isfile(df['indptr']) or not os.path.isfile(df['indices']):
            return {mmd: None for mmd in bin_adj_files}
        if not os.path.isfile(df['data']):
            df['data'] = ''
    return bin_adj_files


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
