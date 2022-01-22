# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc utility functions.

e.g., convert GraphSAINT or OGB data format to shaDow
"""

import json
from graph_engine.frontend import TRAIN, VALID, TEST
from graph_engine.frontend.graph_utils import to_undirected_csr, get_adj_dtype
import scipy.sparse as sp
import os
import glob
import numpy as np
from tqdm import tqdm
import pickle
import itertools
from dataclasses import dataclass
from torch_geometric.utils import to_undirected


@dataclass(frozen=True)
class ShaDowFiles:
    name: str
    source: str
    original_name: str
    prediction_task: str = ""
    required_files: frozenset = frozenset({})

    def have_all_files(self, prefix: str) -> bool:
        """
        Check if all files required by shaDow training are present in the data dir
        """
        return all(
            glob.glob(f"{prefix}/{self.name}/{f}") for f in self.required_files
        )



@dataclass(frozen=True)
class ShaDowFilesNode(ShaDowFiles):
    prediction_task: str = "node"
    required_files: frozenset = frozenset(
        {     # all possible required files
            'adj_full_raw.np[yz]', 
            'adj_train_raw.np[yz]', 
            'label_full.npy', 
            'feat_full.npy', 
            'split.npy' 
        }
    )

@dataclass(frozen=True)
class ShaDowFilesNodeTransductive(ShaDowFilesNode):
    required_files: frozenset = frozenset(
        {
            'adj_full_raw.np[yz]', 
            'label_full.npy', 
            'feat_full.npy', 
            'split.npy' 
        }
    )

@dataclass(frozen=True)
class ShaDowFilesNodeInductive(ShaDowFilesNode):
    pass


@dataclass(frozen=True)
class ShaDowFilesLink(ShaDowFiles):
    prediction_task: str = "link"
    required_files: frozenset = frozenset(
        {
            'adj_full_raw_with_val.np[yz]', 
            'adj_full_raw.np[yz]', 
            'feat_full.npy', 
            'split.npy'
        }
    )
    

@dataclass(frozen=True)
class ShaDowFilesLinkWithValEdges(ShaDowFilesLink):
    required_files: frozenset = frozenset(
        {
            'adj_full_raw_with_val.np[yz]', 
            'feat_full.npy', 
            'split.npy'
        }
    )


@dataclass(frozen=True)
class ShaDowFilesLinkNoValEdges(ShaDowFilesLink):
    required_files: frozenset = frozenset(
        {
            'adj_full_raw.np[yz]', 
            'feat_full.npy', 
            'split.npy'
        }
    )


_node_kSpec_vCls = (
    ('transductive', ShaDowFilesNodeTransductive),
    ('inductive', ShaDowFilesNodeInductive),
    ('ALL', ShaDowFilesNode)
)
_link_kSpec_vCls = (
    ('with_val_edges', ShaDowFilesLinkWithValEdges),
    ('no_val_edges', ShaDowFilesLinkNoValEdges),
    ('ALL', ShaDowFilesLink)
)
DATA_ZOO = {
    "flickr": {
        kSpec: vCls('flickr', 'SAINT', 'flickr')
        for kSpec, vCls in _node_kSpec_vCls
    },
    "reddit": {
        kSpec: vCls('reddit', 'SAINT', 'reddit')
        for kSpec, vCls in _node_kSpec_vCls
    },
    "yelp": {
        kSpec: vCls('yelp', 'SAINT', 'yelp')
        for kSpec, vCls in _node_kSpec_vCls
    },
    "arxiv": {
        kSpec: vCls('arxiv', 'OGB', 'ogbn-arxiv')
        for kSpec, vCls in _node_kSpec_vCls
    },
    "products": {
        kSpec: vCls('products', 'OGB', 'ogbn-products')
        for kSpec, vCls in _node_kSpec_vCls
    },
    "papers100M": {
        kSpec: vCls('papers100M', 'OGB', 'ogbn-papers100M')
        for kSpec, vCls in _node_kSpec_vCls
    },
    "collab": {
        kSpec: vCls("collab", "OGB", "ogbl-collab")
        for kSpec, vCls in _link_kSpec_vCls
    },
    "ppa": {
        kSpec: vCls("ppa", "OGB", "ogbl-ppa")
        for kSpec, vCls in _link_kSpec_vCls
    }
}


FILES_SAINT  = frozenset(
    {
        'adj_full.npz', 
        'adj_train.npz', 
        'feats.npy', 
        'class_map.json', 
        'role.json'
    }
)


def _convert_saint2shadow(data_meta, dir_shadow: str, dir_saint: str) -> None:
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
    np.save(
        dir_shadow.format('split.npy'), 
        {
            TRAIN: np.asarray(role['tr'], dtype=dtype),
            VALID: np.asarray(role['va'], dtype=dtype),
            TEST : np.asarray(role['te'], dtype=dtype)
        }
    )    
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


def _convert_ogb2shadow_node(data_meta, dir_shadow: str, dir_ogb: str) -> None:
    """
    For node classification tasks: convert from OGB format to shaDow-GNN format
    """
    from ogb.nodeproppred import PygNodePropPredDataset
    print(f"Preparing shaDow-GNN 'node' dataset from OGB format")
    dir_ogb_parent = '/'.join(dir_ogb.split('/')[:-1])
    if not os.path.exists(dir_ogb_parent):
        os.makedirs(dir_ogb_parent)
    dataset = PygNodePropPredDataset(data_meta.original_name, root=dir_ogb_parent)
    split_idx = dataset.get_idx_split()
    graph = dataset[0]
    num_node = graph.y.shape[0]
    num_edge = graph.edge_index.shape[1]
    # feat_full.npy
    np.save(dir_shadow.format('feat_full.npy'), graph.x.numpy().astype(np.float32, copy=False))
    graph.x = None          # done with x, so dereference the pointer to save some memory
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
    for _k, v in split_idx.items():
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
    adj_full = graph = None
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
    np.save(
        dir_shadow.format('split.npy'), 
        {
            TRAIN: split_idx['train'].numpy().astype(dtype, copy=False),
            VALID: split_idx['valid'].numpy().astype(dtype, copy=False),
            TEST : split_idx['test'].numpy().astype(dtype, copy=False)
        }
    )
    print(f"Successfully saved shaDow-GNN dataset into {'/'.join(dir_shadow.split('/')[:-1])}")
    

def _convert_ogb2shadow_link(data_meta, dir_shadow: str, dir_ogb: str) -> None:
    """
    For link prediction tasks: convert from OGB format to shaDow-GNN format
    """
    name_data = data_meta.name
    from ogb.linkproppred import PygLinkPropPredDataset
    print(f"Preparing shaDow-GNN 'link' dataset from OGB format")
    dir_ogb_parent = '/'.join(dir_ogb.split('/')[:-1])
    if not os.path.exists(dir_ogb_parent):
        os.makedirs(dir_ogb_parent)
    dataset = PygLinkPropPredDataset(data_meta.original_name, root=dir_ogb_parent)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    num_node = graph.x.shape[0]
    assert num_node == graph.num_nodes    
    # feat_full.npy
    np.save(dir_shadow.format('feat_full.npy'), graph.x.numpy().astype(np.float32, copy=False))
    graph.x = None        
    if name_data == 'collab':
        """
        split_edge: 
            train   edge/weight/year
            valid   edge/weight/year/edge_neg
            test    edge/weight/year/edge_neg
        where edge & edge_neg are 2D: m x 2; and weight & year are 1D: m
            leading dim of edge & weight & year are the same
        split_edge['train']['edge'].shape[0] + split_edge['valid']['edge'].shape[0] + split_edge['test']['edge'][0]
            matches the edge number in ogb paper
        """
        assert graph.edge_index.shape[1] == graph.edge_weight.shape[0] == graph.edge_year.shape[0]
        """
        adj -- prepare two versions
            in the vanilla setting, adj_full only contains edges in the training set
            in the alternative setting, adj_full contains validation edges as well (i.e., use_valedges_as_input)
        row_full, col_full = graph.edge_index.numpy()
        By default we perform coalescing and store in adj_full_raw. 
            without coalescing, there will be multiple edges between nodes, and thus csr is invalid
        """
        from torch_sparse import coalesce
        edge_index, edge_weight = coalesce(graph.edge_index, graph.edge_weight, num_node, num_node)
        row_full, col_full = edge_index.numpy()
        adj_full = sp.coo_matrix(
            (
                edge_weight.numpy().flatten(),
                (row_full, col_full),
            ), shape=(num_node, num_node)
        ).tocsr()
        dtype = get_adj_dtype(adj=adj_full)
        adj_full.indptr = adj_full.indptr.astype(dtype, copy=False)
        adj_full.indices = adj_full.indices.astype(dtype, copy=False)
        sp.save_npz(dir_shadow.format('adj_full_raw.npz'), adj_full)
        adj_full = None
        # valedge as input
        valedges_und = to_undirected(split_edge['valid']['edge'].t()).numpy()
        row_train_val, col_train_val = np.concatenate([graph.edge_index.numpy(), valedges_und], axis=1)
        edge_weight_train_val = np.concatenate(
            [graph.edge_weight.numpy().flatten(), np.ones(valedges_und.shape[1])]
        )
        adj_full_train_val = sp.coo_matrix(
            (
                edge_weight_train_val,
                (row_train_val, col_train_val),
            ), shape=(num_node, num_node)
        ).tocsr()
        dtype = get_adj_dtype(adj=adj_full_train_val)
        adj_full_train_val.indptr = adj_full_train_val.indptr.astype(dtype, copy=False)
        adj_full_train_val.indices = adj_full_train_val.indices.astype(dtype, copy=False)
        sp.save_npz(dir_shadow.format('adj_full_raw_with_val.npz'), adj_full_train_val)
        adj_full_train_val = None
        graph = None
        # skip adj_train for link task --> current don't consider inductive link prediction
        # split.npy     --> positive and negative sample of edges
        np.save(
            dir_shadow.format('split.npy'), 
            {
                TRAIN: {'pos': split_edge['train']['edge'].numpy().astype(dtype, copy=False)},
                VALID: {'pos': split_edge['valid']['edge'].numpy().astype(dtype, copy=False),
                        'neg': split_edge['valid']['edge_neg'].numpy().astype(dtype, copy=False)},
                TEST : {'pos': split_edge['test']['edge'].numpy().astype(dtype, copy=False),
                        'neg': split_edge['test']['edge_neg'].numpy().astype(dtype, copy=False)}
                # 'ALL': {'pos': graph.edge_index.numpy().astype(dtype, copy=False)}
            }
        )
    elif name_data == 'ppa':
        row, col = graph.edge_index
        adj_full = sp.coo_matrix(
            (
                np.ones(graph.num_edges),
                (row.numpy(), col.numpy()),
            ), shape=(num_node, num_node)
        ).tocsr()
        dtype = get_adj_dtype(adj=adj_full)
        adj_full.indptr = adj_full.indptr.astype(dtype, copy=False)
        adj_full.indices = adj_full.indices.astype(dtype, copy=False)
        sp.save_npz(dir_shadow.format('adj_full_raw.npz'), adj_full)
        adj_full = graph = None
        # same as collab
        np.save(
            dir_shadow.format('split.npy'), 
            {
                TRAIN: {'pos': split_edge['train']['edge'].numpy().astype(dtype, copy=False)},
                VALID: {'pos': split_edge['valid']['edge'].numpy().astype(dtype, copy=False),
                        'neg': split_edge['valid']['edge_neg'].numpy().astype(dtype, copy=False)},
                TEST : {'pos': split_edge['test']['edge'].numpy().astype(dtype, copy=False),
                        'neg': split_edge['test']['edge_neg'].numpy().astype(dtype, copy=False)}
                # 'ALL': {'pos': graph.edge_index.numpy().astype(dtype, copy=False)}
            }
        )
    else:
        raise NotImplementedError
    print(f"Successfully saved shaDow-GNN dataset into {'/'.join(dir_shadow.split('/')[:-1])}")


def convert2shaDow(
    name_data: str, 
    prefix: str, 
    specification_task: str="ALL"
) -> None:
    if not os.path.exists(f"{prefix}/{name_data}"):
        os.makedirs(f"{prefix}/{name_data}")
    dir_shadow = f"{prefix}/{name_data}/" + "{}"
    dir_saint  = f"{prefix}/saint/{name_data}/" + "{}"
    dir_ogb    = f"{prefix}/ogb/{name_data}/" + "{}"
    _DATA_META = DATA_ZOO[name_data][specification_task]
    prediction_task = _DATA_META.prediction_task
    source = _DATA_META.source
    if _DATA_META.have_all_files(prefix):
        pass
    elif source == "OGB":
        if prediction_task == "node":
            _convert_ogb2shadow_node(_DATA_META, dir_shadow, dir_ogb)
        else:
            _convert_ogb2shadow_link(_DATA_META, dir_shadow, dir_ogb)
    elif source == "SAINT":
        assert all(glob.glob(dir_saint.format(f)) for f in FILES_SAINT)
        _convert_saint2shadow(_DATA_META, dir_shadow, dir_saint)
    else:
        raise NotImplementedError
    _precompute_data(name_data, dir_shadow)
    return


def _precompute_data(name_data: str, dir_shadow: str) -> None:
    """
    Save the undirected version of the adj on disk. This saves some time for 
    large datasets such as papers100M.
    """
    if name_data in ['papers100M', 'arxiv']:
        adj_full = sp.load_npz(dir_shadow.format('adj_full_raw.npz'))
        adj_train = sp.load_npz(dir_shadow.format('adj_train_raw.npz'))
        adj_full_und = to_undirected_csr(adj_full)
        adj_train_und = to_undirected_csr(adj_train)
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
    elif name_data == 'collab':
        adj_full = sp.load_npz(dir_shadow.format('adj_full_raw.npz'))
        adj_full_valedges = sp.load_npz(dir_shadow.format('adj_full_raw_with_val.npz'))
        adj_full_und = to_undirected_csr(adj_full)
        adj_full_und_valedges = to_undirected_csr(adj_full_valedges)
        with open(dir_shadow.format('adj_full_undirected.npy'), 'wb') as f:
            pickle.dump({'indptr': adj_full_und.indptr, 'indices': adj_full_und.indices}, f, protocol=4)
        with open(dir_shadow.format('adj_full_undirected_with_val.npy'), 'wb') as f:
            pickle.dump({'indptr': adj_full_und_valedges.indptr, 'indices': adj_full_und_valedges.indices}, f, protocol=4)
    elif name_data == 'ppa':
        adj_full = sp.load_npz(dir_shadow.format('adj_full_raw.npz'))
        adj_full_und = to_undirected_csr(adj_full)
        with open(dir_shadow.format('adj_full_undirected.npy'), 'wb') as f:
            pickle.dump({'indptr': adj_full_und.indptr, 'indices': adj_full_und.indices}, f, protocol=4)


def cleanup_shaDow(name_data: str, prefix: str) -> None:
    """delete the converted shaDow dataset. 
    Policy: safe deletion
        Only delete shaDow files when 
        1. the raw data (e.g., in SAINT format) exists. 
        2. the files in the folder are exactly the set of shaDow files
    """
    if name_data not in DATA_ZOO:
        raise RuntimeError(f"data {name_data} is not supported")
    dir_shadow = f"{prefix}/{name_data}/"
    candy_files = set(next(os.walk(dir_shadow))[-1])
    _DATA_META = DATA_ZOO[name_data]['ALL']
    all_allowed_files = itertools.chain(
        *[glob.glob(f'{dir_shadow}/{f}') for f in _DATA_META.required_files]
    )
    all_allowed_files = set(f.split('/')[-1] for f in all_allowed_files)
    if len(all_allowed_files) < len(candy_files):
        raise RuntimeError(
            "shaDow data dir possibly corrupted: contains extra files not belonging to shaDow data\nAborting!"
        )
    if _DATA_META.source == 'SAINT':
        dir_saint = f"{prefix}/saint/{name_data}"
        if not os.path.isdir(dir_saint):
            raise RuntimeError("raw data in SAINT format does not exist. \nAborting!")
        raw_candy_files = set(next(os.walk(dir_saint))[-1])
        if FILES_SAINT != raw_candy_files:
            raise RuntimeError("raw data in saint/ dir does not match the SAINT format\nAborting!")
    for f in candy_files:
        assert os.path.isfile(f"{dir_shadow}/{f}"), f"shaDow data file {dir_shadow}/{f} does not exist!"
        os.remove(f"{dir_shadow}/{f}")
        print(f"Successfully removed file {dir_shadow}/{f}")
    