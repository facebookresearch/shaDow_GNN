# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from shaDow import TRAIN, VALID, TEST
import scipy.sparse as sp
import os
import glob
import numpy as np
from tqdm import tqdm
import pickle

def get_adj_dtype(adj=None, num_nodes=-1, num_edges=-1):
    if adj is not None:
        num_nodes, num_edges = adj.shape[0], adj.size
    else:
        assert num_edges > 0 and num_nodes > 0
    return np.uint32 if max(num_nodes, num_edges) < 2**32 else np.int64

def _convert_saint2shadow(name_data, dir_shadow, dir_saint):
    print(f"Preparing shaDow-GNN dataset from GraphSAINT format")
    adj_full = sp.load_npz(dir_saint.format('adj_full.npz'))
    dtype = get_adj_dtype(adj=adj_full)
    # adj_full.npz -> adj_full_raw.npz
    if adj_full.data.min() == adj_full.data.max() == 1.:
        adj_f_data = np.broadcast_to(np.ones(1, dtype=np.bool), adj_full.data.size)
    else:
        adj_f_data = adj_full.data.astype(np.float32, copy=False)
    adj_f_indptr = adj_full.indptr
    adj_f_indices = adj_full.indices
    adj_ = sp.csr_matrix((adj_f_data, adj_f_indices, adj_f_indptr), shape=adj_full.shape)
    adj_.indptr = adj_.indptr.astype(dtype, copy=False)
    adj_.indices = adj_.indices.astype(dtype, copy=False)
    sp.save_npz(dir_shadow.format('adj_full_raw.npz'), adj_)
    # adj_train.npz -> adj_train_raw.npz
    adj_train = sp.load_npz(dir_saint.format('adj_train.npz'))
    if adj_train.data.min() == adj_train.data.max() == 1:
        adj_t_data = np.broadcast_to(np.ones(1, dtype=np.bool), adj_train.data.size)
    else:
        adj_t_data = adj_train.data.astype(np.float32, copy=False)
    adj_t_indptr = adj_train.indptr
    adj_t_indices = adj_train.indices
    adj_ = sp.csr_matrix((adj_t_data, adj_t_indices, adj_t_indptr), shape=adj_train.shape)
    adj_.indptr = adj_.indptr.astype(dtype, copy=False)
    adj_.indices = adj_.indices.astype(dtype, copy=False)
    sp.save_npz(dir_shadow.format('adj_train_raw.npz'), adj_)
    # role.json -> split.npy
    with open(dir_saint.format('role.json')) as fr:
        role = json.load(fr)
    np.save(dir_shadow.format('split.npy'), {
                TRAIN: np.asarray(role['tr'], dtype=dtype),
                VALID: np.asarray(role['va'], dtype=dtype),
                TEST : np.asarray(role['te'], dtype=dtype)})    
    # class_map.json -> label_full.npy
    with open(dir_saint.format('class_map.json')) as fc:
        class_map = json.load(fc)
    class_map = {int(k): v for k, v in class_map.items()}
    num_nodes = adj_full.shape[0]
    class_val_0 = next(iter(class_map.values()))
    if isinstance(class_val_0, list):
        num_classes = len(class_val_0)
        label_full = np.zeros((num_nodes, num_classes), dtype=np.bool)
        for k, v in class_map.items():
            label_full[k] = v
    else:       # class label is represented as an int
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        label_full = np.zeros((num_nodes, num_classes), dtype=np.bool)
        offset = min(class_map.values())
        idx0 = np.asarray(list(class_map.keys()))
        idx1 = np.asarray(list(class_map.values())) - offset
        label_full[idx0, idx1] = 1
    np.save(dir_shadow.format('label_full.npy'), label_full)
    # feats.npy -> feat_full.npy
    feats = np.load(dir_saint.format('feats.npy'))
    np.save(dir_shadow.format('feat_full.npy'), feats.astype(np.float32, copy=False))
    print(f"Successfully saved shaDow-GNN dataset into {'/'.join(dir_shadow.split('/')[:-1])}")


def _convert_ogb2shadow(name_data, dir_shadow, dir_ogb):
    from ogb.nodeproppred import PygNodePropPredDataset
    print(f"Preparing shaDow-GNN dataset from OGB format")
    name_map_shadow2ogb = {'arxiv': 'ogbn-arxiv', 'products': 'ogbn-products', 'papers100M': 'ogbn-papers100M'}
    dir_ogb_parent = '/'.join(dir_ogb.split('/')[:-1])
    if not os.path.exists(dir_ogb_parent):
        os.makedirs(dir_ogb_parent)
    dataset = PygNodePropPredDataset(name_map_shadow2ogb[name_data], root=dir_ogb_parent)
    split_idx = dataset.get_idx_split()
    graph = dataset[0]
    num_node = graph.y.shape[0]
    num_edge = graph.edge_index.shape[1]
    # feat_full.npy
    np.save(dir_shadow.format('feat_full.npy'), graph.x.numpy().astype(np.float32, copy=False))
    graph.x = None
    # label_full.npy        NOTE only for single class classification. Otherwise, cannot use 1D label arr
    y_non_nan = graph.y[graph.y == graph.y]
    assert y_non_nan.min().item() == 0
    if y_non_nan.max().item() < 2**8:
        dtype_l = np.uint8
    elif y_non_nan.max().item() < 2**16:
        dtype_l = np.uint16
    elif y_non_nan.max().item() < 2**32:   # Almost impossible to have so many classes
        dtype_l = np.uint32
    else:
        dtype_l = np.int64
    # assert all train / valid / test nodes are not nan
    for k, v in split_idx.items():
        assert not graph.y[v].isnan().any().item()
    np.save(dir_shadow.format('label_full.npy'), graph.y.numpy().flatten().astype(dtype_l, copy=False))
    # adj_full_raw.npz
    row_full, col_full = graph.edge_index.numpy()
    adj_full = sp.coo_matrix(
        (
            np.broadcast_to(np.ones(1, dtype=np.bool), row_full.size),
            (row_full, col_full),
        ),
        shape=(num_node, num_node)
    ).tocsr()
    dtype = get_adj_dtype(adj=adj_full)
    adj_full.indptr = adj_full.indptr.astype(dtype, copy=False)
    adj_full.indices = adj_full.indices.astype(dtype, copy=False)
    sp.save_npz(dir_shadow.format('adj_full_raw.npz'), adj_full)
    adj_full = None
    # adj_train_raw.npz
    idx_train_set = set(split_idx['train'].numpy().tolist())
    idx_test_set = set(split_idx['test'].numpy().tolist())
    row_train, col_train = [], []
    print("Converting adj into the shaDow format")
    for i in tqdm(range(row_full.size)):
        if row_full[i] in idx_train_set and col_full[i] in idx_train_set:
            row_train.append(row_full[i])
            col_train.append(col_full[i])
    adj_train = sp.coo_matrix(
        (
            np.broadcast_to(np.ones(1, dtype=np.bool), len(row_train)),
            (np.asarray(row_train), np.asarray(col_train)),
        ),
        shape=(num_node, num_node)
    ).tocsr()
    row_train = col_train = None
    adj_train.indptr = adj_train.indptr.astype(dtype, copy=False)
    adj_train.indices = adj_train.indices.astype(dtype, copy=False)
    sp.save_npz(dir_shadow.format('adj_train_raw.npz'), adj_train)
    adj_train = None
    # split.npy (need to do as the last step, since dtype should be determined by adj_full)
    np.save(dir_shadow.format('split.npy'), {
                TRAIN: split_idx['train'].numpy().astype(dtype, copy=False),
                VALID: split_idx['valid'].numpy().astype(dtype, copy=False),
                TEST : split_idx['test'].numpy().astype(dtype, copy=False)})
    print(f"Successfully saved shaDow-GNN dataset into {'/'.join(dir_shadow.split('/')[:-1])}")
    

def convert2shaDow(name_data, prefix):
    if not os.path.exists(f"{prefix}/{name_data}"):
        os.makedirs(f"{prefix}/{name_data}")
    dir_shadow = f"{prefix}/{name_data}/" + "{}"
    dir_saint  = f"{prefix}/saint/{name_data}/" + "{}"
    dir_ogb    = f"{prefix}/ogb/{name_data}/" + "{}"
    fs_shadow = ['adj_full_raw.np[yz]', 'adj_train_raw.np[yz]', 'label_full.npy', 'feat_full.npy', 'split.npy']
    fs_saint  = ['adj_full.npz', 'adj_train.npz', 'feats.npy', 'class_map.json', 'role.json']
    if all(glob.glob(dir_shadow.format(f)) for f in fs_shadow):
        pass
    elif name_data in ['products', 'arxiv', 'papers100M']:
        _convert_ogb2shadow(name_data, dir_shadow, dir_ogb)
    else:
        assert all(glob.glob(dir_saint.format(f)) for f in fs_saint)
        _convert_saint2shadow(name_data, dir_shadow, dir_saint)
    precompute_data(name_data, dir_shadow)
    return


def precompute_data(name_data, dir_shadow):
    """
    Save the undirected version of the adj on disk. This saves some time for 
    large datasets such as papers100M.
    """
    if name_data in ['papers100M', 'arxiv']:
        adj_full = sp.load_npz(dir_shadow.format('adj_full_raw.npz'))
        adj_train = sp.load_npz(dir_shadow.format('adj_train_raw.npz'))
        adj_full_und = to_undirected(adj_full)
        adj_train_und = to_undirected(adj_train)
        with open(dir_shadow.format('adj_full_undirected.npy'), 'wb') as f:
            pickle.dump({'indptr': adj_full_und.indptr, 'indices': adj_full_und.indices}, f, protocol=4)
        with open(dir_shadow.format('adj_train_undirected.npy'), 'wb') as f:
            pickle.dump({'indptr': adj_train_und.indptr, 'indices': adj_train_und.indices}, f, protocol=4)
        if not os.path.exists(dir_shadow.format('cpp')):
            os.makedirs(dir_shadow.format('cpp'))
        fm_cpp_bin = dir_shadow.format('cpp') + '/adj_{}_{}_{}.bin'
        adj_full_und.indptr.tofile(fm_cpp_bin.format('full', 'undirected', 'indptr'))
        adj_full_und.indices.tofile(fm_cpp_bin.format('full', 'undirected', 'indices'))
        adj_train_und.indptr.tofile(fm_cpp_bin.format('train', 'undirected', 'indptr'))
        adj_train_und.indices.tofile(fm_cpp_bin.format('train', 'undirected', 'indices'))


def to_undirected(adj):
    """
    Convert a directed graph into undirected.

    input adj is in csr format
    returned adj is in csr format
    """
    print("Converting graph to undirected. This may take a while for large graphs ...")
    adj_coo = adj.tocoo()
    adj_trans = sp.coo_matrix((adj_coo.data, (adj_coo.col, adj_coo.row)), shape=adj_coo.shape)
    adj_trans = adj_trans.tocsr()
    indptr_new = np.zeros(adj.indptr.size, dtype=np.int64)
    indices_new = []
    for i in range(adj.shape[0]):
        neigh1 = adj.indices[adj.indptr[i] : adj.indptr[i + 1]]
        neigh2 = adj_trans.indices[adj_trans.indptr[i] : adj_trans.indptr[i + 1]]
        neigh_merged = np.union1d(neigh1, neigh2)
        indptr_new[i + 1] = indptr_new[i] + neigh_merged.size
        indices_new.append(neigh_merged)
    indices_new = np.concatenate(indices_new)
    assert adj.data.max() == adj.data.min() == 1
    data_new = np.broadcast_to(np.ones(1, dtype=np.bool), indices_new.size)
    adj_und = sp.csr_matrix((data_new, indices_new, indptr_new), shape=adj.shape)
    dtype = get_adj_dtype(adj=adj_und)
    adj_und.indptr = adj_und.indptr.astype(dtype, copy=False)
    adj_und.indices = adj_und.indices.astype(dtype, copy=False)
    return adj_und
