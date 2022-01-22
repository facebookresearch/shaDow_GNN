# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.sparse as sp
import torch
import os
from collections import defaultdict
from graph_engine.frontend.data_converter import convert2shaDow, DATA_ZOO
from graph_engine.frontend.graph_utils import to_undirected_csr
from graph_engine.frontend import TRAIN, VALID, TEST
from graph_engine.frontend.graph import RawGraph
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any


def load_data(
    prefix: Dict[str, str], 
    dataset: str, 
    config_data: Dict[str, Any], 
    printf=lambda txt, style=None: print(txt)
) -> RawGraph:
    """
    valedges_as_input       only applicable to the link task of collab

    Return:
        Graph dataclass
    """
    prediction_task = DATA_ZOO[dataset]["ALL"].prediction_task
    printf("Loading training data..")
    prefix_l = prefix['local']
    surfix_adj_f = ''
    if prediction_task == 'link':
        assert 'transductive' not in config_data or config_data['transductive']
        config_data['transductive'] = True
        spec_task = "with_val_edges" if config_data["valedges_as_input"] else "no_val_edges"
    else:
        spec_task = "transductive" if config_data['transductive'] else "inductive"
    assert prediction_task == DATA_ZOO[dataset][spec_task].prediction_task
    # download and convert raw data if not done so already.
    if not DATA_ZOO[dataset][spec_task].have_all_files(prefix_l):
        convert2shaDow(dataset, prefix_l)
    # load data in shaDow format
    role = np.load(f"{prefix_l}/{dataset}/split.npy", allow_pickle=True)
    if type(role) == np.ndarray:
        role = role[()]
    else:
        assert type(role) == dict
    if prediction_task == 'node':        
        # role is used as index, which is required to be int64 (node_set won't take much mem anyways)
        node_set = {}
        for k in [TRAIN, VALID, TEST]:
            node_set[k] = np.asarray(role[k], dtype=np.int64)
        label_full = torch.from_numpy(np.load(f"{prefix_l}/{dataset}/label_full.npy"))
    else:
        edge_set = defaultdict(dict)
        for k in role.keys():
            for k2, v2 in role[k].items():
                edge_set[k][k2] = np.asarray(v2, dtype=np.int64)
    # load adj. If we want to convert to_undirected, and the undirected adj has been stored as external file,
    # then we skip the conversion in the program and directly load the undirected adj. 
    bin_adj_files = {
        md: {'indptr': None, 'indices': None, 'data': None} for md in [TRAIN, VALID, TEST]
    }
    def fill_bin_adj_dict(mode_, split_, type_, surfix=''):
        for d in ['indptr', 'indices', 'data']:
            bin_adj_files[mode_][d] = f"{prefix_l}/{dataset}/cpp/adj_{split_}_{type_}_{d}{surfix}.bin"
    if 'coalesce' in config_data and not config_data['coalesce']:
        raise NotImplementedError
    if config_data['to_undirected']:
        if (adj_full := _load_adj(prefix_l, dataset, 'undirected', 'full', surfix=surfix_adj_f)) is None:
            adj_full = _load_adj(prefix_l, dataset, 'raw', 'full', surfix=surfix_adj_f)
            adj_full = to_undirected_csr(adj_full)
        for md in (VALID, TEST):
            fill_bin_adj_dict(md, 'full', 'undirected', surfix=surfix_adj_f)
        if config_data['transductive']:
            adj_train = adj_full
            fill_bin_adj_dict(TRAIN, 'full', 'undirected', surfix=surfix_adj_f)
        elif (adj_train := _load_adj(prefix_l, dataset, 'undirected', 'train', surfix=surfix_adj_f)) is None:
            adj_train = _load_adj(prefix_l, dataset, 'raw', 'train', surfix=surfix_adj_f)
            adj_train = to_undirected_csr(adj_train)
            fill_bin_adj_dict(TRAIN, 'train', 'undirected', surfix=surfix_adj_f)
            assert set(adj_train.indices).issubset(set(node_set[TRAIN]))
    else:
        adj_full = _load_adj(prefix_l, dataset, 'raw', 'full', surfix=surfix_adj_f)
        for md in (VALID, TEST):
            fill_bin_adj_dict(md, 'full', 'raw', surfix=surfix_adj_f)
        if config_data['transductive']:
            adj_train = adj_full
            fill_bin_adj_dict(TRAIN, 'full', 'raw', surfix=surfix_adj_f)
        else:
            adj_train = _load_adj(prefix_l, dataset, 'raw', 'train', surfix=surfix_adj_f)
            assert set(adj_train.nonzero()[0]).issubset(set(node_set[TRAIN]))
            fill_bin_adj_dict(TRAIN, 'train', 'raw', surfix=surfix_adj_f)
    bin_adj_files = validate_bin_file(bin_adj_files)

    printf(f"SETTING TO {'TRANS' if config_data['transductive'] else 'IN'}DUCTIVE LEARNING", style="red")
    
    # ======= deal with feats =======
    mode_norm = 'all' if config_data['transductive'] else 'train'
    if config_data['norm_feat'] and os.path.isfile(f"{prefix_l}/{dataset}/feat_full_norm_{mode_norm}.npy"):
        feats = np.load(f"{prefix_l}/{dataset}/feat_full_norm_{mode_norm}.npy")
        printf(f"Loading '{mode_norm}'-normalized features", style='yellow')
    else:
        feats = np.load(f"{prefix_l}/{dataset}/feat_full.npy")
        if config_data['norm_feat']:
            feats_fit = feats if config_data['transductive'] else feats[node_set[TRAIN]]
            scaler = StandardScaler()
            scaler.fit(feats_fit)
            feats = scaler.transform(feats)
            printf(f"Normalizing node features (mode = {mode_norm})", style="yellow")
        else:
            printf("Not normalizing node features", style="yellow")
    feats = torch.from_numpy(feats).type(torch.get_default_dtype())
    printf("Done loading training data..")
    # return
    if prediction_task == 'node':
        return RawGraph(adj_full, adj_train, feats, label_full, node_set, None, bin_adj_files)
    else:
        return RawGraph(adj_full, adj_train, feats, None, None, edge_set, bin_adj_files)


def _load_adj(prefix, dataset, type_, split_, surfix=''):
    """
    Try to load the prestored undirected adj. If the file does not exist, then you MUST return a None
    """
    assert split_ in ['full', 'train'], "UNKNOWN ADJ SPLIT. ONLY ACCEPT [full] or [train]"
    assert type_ in ['raw', 'undirected'], "UNKNOWN ADJ TYPE. ONLY ACCEPT [raw] or [undirected]"
    file_adj = f"{prefix}/{dataset}/adj_{split_}_{type_}{surfix}." + "{}"
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
    for _md, df in bin_adj_files.items():
        assert set(df.keys()) == set(['indptr', 'indices', 'data'])
        if not os.path.isfile(df['indptr']) or not os.path.isfile(df['indices']):
            return {mmd: None for mmd in bin_adj_files}
        if not os.path.isfile(df['data']):
            df['data'] = ''
    return bin_adj_files
