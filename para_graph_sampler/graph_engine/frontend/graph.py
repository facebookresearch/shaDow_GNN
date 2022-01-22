from dataclasses import dataclass, InitVar
import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, Optional, Union
import torch


@dataclass
class RawGraph:
    """
    data struct representing the original full graph without
    any sampling operation. Can be used for both node-classification
    and link-prediction. 

    TODO: support graph classification. 
    """
    adj_full: sp.csr.csr_matrix
    adj_train: Optional[sp.csr.csr_matrix]
    feat_full: Union[np.ndarray, torch.tensor, None]
    label_full: Union[np.ndarray, torch.tensor, None]
    node_set: Optional[Dict[Any, Union[np.ndarray, torch.tensor]]]     # TODO: merge node_set and edge_set into a single entity_set
    edge_set: Optional[Dict[Any, Union[np.ndarray, torch.tensor]]]
    bin_adj_files: Optional[Dict[Any, Optional[str]]]

    def __post_init__(self):
        self._validate()
    
    @property
    def entity_set(self):
        if self.node_set is None:
            return self.edge_set
        return self.node_set

    @property
    def num_nodes(self):
        return self.adj_full.indptr.size - 1

    @property
    def num_edges(self):
        return self.adj_full.indices.size
    
    def _validate(self):
        """
        TODO: now only based on homogeneous graph, and thus we have a simple csr. 
        To be extended to heterogeneous setting in the near future. 
        """
        if self.feat_full is not None:
            assert self.feat_full.shape[0] == self.num_nodes, \
                f"[RawGraph]: unmatched feature size ({self.feat_full.shape[0]}) and graph size ({self.num_nodes})"
        if self.label_full is not None:
            assert self.label_full.shape[0] == self.num_nodes, \
                f"[RawGraph]: unmatched label size ({self.label_full.shape[0]}) and graph size ({self.num_nodes})"
    
    def deinit(self):
        """
        de-reference everything (e.g., tensors, numpy), so that the memory can be freed by GC
        """
        for field in self.__dataclass_fields__:
            setattr(self, field, None)


@dataclass
class EntityEncoding:
    hop             : np.ndarray = np.array([])
    ppr             : np.ndarray = np.array([])
    drnl            : np.ndarray = np.array([])
    # summary: need to be ordered and immutable
    names_data_fields = ('hop', 'ppr', 'drnl')
    # init fields
    cap_node_subg   : InitVar[int] = None
    cap_edge_subg   : InitVar[int] = None
    validate        : InitVar[bool] = True

    def __post_init__(self, cap_node_subg, cap_edge_subg, validate: bool):
        if cap_node_subg is not None and cap_edge_subg is not None:
            dtype = {
                'hop'   : np.int64,
                'ppr'   : np.float32,
                'drnl'  : np.int64
            }
            f_dtype = lambda n : np.uint16 if n < 2**16 else np.uint32
            if cap_node_subg < 2**32:
                dtype['hop'] = f_dtype(cap_node_subg)
                dtype['drnl'] = f_dtype(cap_node_subg**2)
        
        if validate:
            self.check_valid()
    
    def check_valid(self):
        enc_len = {getattr(self, n).size for n in self.names_data_fields}
        if 0 in enc_len:
            enc_len.remove(0)
        assert len(enc_len) <= 1, 'all entity enc should have the same length (num subg nodes)'
    
    def check_valid_(self, subg):
        assert self.hop.size == 0 or self.hop.size == subg.num_nodes
        assert self.ppr.size == 0 or self.ppr.size == subg.num_nodes
        assert self.drnl.size == 0 or self.drnl.size == subg.num_nodes
    
    def fill_hops(self, subg):
        """
        Only used by python sampler. For C++ sampler, the backend will take care of the "hops" annotation. 
        Set the hop number for all subgraph nodes
        
        Update the values of self.hop
        """
        assert subg.target.size == 1, 'use drnl or other de for feat augmentation'
        node2hop = {n: None for n in range(subg.indptr.size - 1)}
        node2hop[subg.target[0]] = 0
        cur_level = set(subg.target)
        num_level = 0
        next_level = set()
        while len(cur_level) > 0:
            next_level = set()
            num_level += 1
            for n in cur_level:
                for u in subg.indices[subg.indptr[n] : subg.indptr[n + 1]]:
                    if node2hop[u] is not None:
                        continue
                    node2hop[u] = num_level
                    next_level.add(u)
            cur_level = next_level
        assert node2hop[subg.target[0]] == 0
        self.hop = np.fromiter(node2hop.values(), dtype=np.int)
    
    def fill_drnl(self):
        raise NotImplementedError

    def hop2onehot_vec(self, dim_1hot_vec: int, return_type: str="tensor"):
        """
        1-hot encoding to facilitate NN input
        
        dim_1hot_vec = max hop to keep + 0-hop (self) + infty-hop (unreachable)
        """
        assert len(self.hop.shape) == 1
        ret = np.zeros((self.hop.size, dim_1hot_vec))
        valid_h = [-1, 0] + [i for i in range(1, dim_1hot_vec - 1)]
        for i in valid_h:
            ret[np.where(self.hop == i)[0], i + 1] = 1
        # handle overflow separately
        ret[np.where(self.hop >= 255)[0], 0] = 1
        return torch.tensor(ret) if return_type == 'tensor' else ret

    def ppr2onehot_vec(self, dim_1hot_vec: int, return_type: str="tensor"):
        """1-hot encoding to facilitate NN input"""
        assert len(self.ppr.shape) == 1
        ret = np.zeros((self.ppr.size, dim_1hot_vec))
        # below just a very rough heuristic...
        cond_filter = [0.25**i for i in range(dim_1hot_vec)]
        cond_filter += [0]
        for i in range(dim_1hot_vec):
            ret[np.where(np.logical_and(self.ppr <= cond_filter[i], self.ppr >= cond_filter[i+1])), i] = 1
        return torch.tensor(ret) if return_type == 'tensor' else ret

    def drnl2onehot_vec(self, dim_1hot_vec: int, return_type: str="tensor"):
        """
        1-hot encoding to facilitate NN input
        
        dim_1hot_vec = max drnl to keep + infty-hop (unreachable)
        """
        assert len(self.drnl.shape) == 1
        self.drnl[self.drnl >= 255] = 0
        self.drnl[self.drnl > dim_1hot_vec - 1] = 0
        self.drnl[self.drnl < 0] = 0
        ret = np.zeros((self.drnl.size, dim_1hot_vec))      # idx 0 = unreachable by one of the targets. i.e., infty drnl
        ret[np.arange(self.drnl.size), self.drnl] = 1
        return torch.tensor(ret) if return_type == 'tensor' else ret
    
    @classmethod
    def cat_batch(cls, subgs_batch):
        if subgs_batch[0].entity_enc.ppr.size == 0:
            ppr_batch = np.array([])
        else:
            ppr_batch = np.concatenate([s.entity_enc.ppr for s in subgs_batch])
        hop_batch = np.concatenate([s.entity_enc.hop for s in subgs_batch])
        drnl_batch = np.concatenate([s.entity_enc.drnl for s in subgs_batch])
        return cls(hop=hop_batch, ppr=ppr_batch, drnl=drnl_batch)


@dataclass
class ProfileSubgraph:
    pass
    # hops stats
    
    def get_neighbor_hop_composition(self):
        pass


@dataclass
class Subgraph:
    """
    Represents the meta information of sampled subgraphs. 
    """
    # TODO merge the classes of Subgraph and OneBatchSubgraph
    # data fields
    indptr          : np.ndarray
    indices         : np.ndarray
    data            : np.ndarray
    node            : np.ndarray = np.array([])
    edge_index      : np.ndarray = np.array([])
    target          : np.ndarray = np.array([])
    # label
    # size_subg
    # node = idx_raw??    change node to node_index
    # node_feature
    entity_enc      : Optional[EntityEncoding] = None
    # TODO store subgraph profiling data
    # init fields
    cap_node_full   : InitVar[int] = None
    cap_edge_full   : InitVar[int] = None
    cap_node_subg   : InitVar[int] = None
    cap_edge_subg   : InitVar[int] = None
    validate        : InitVar[bool] = True
    
    # summary: need to be ordered and immutable
    names_data_fields = ('indptr', 'indices', 'data', 'node', 'edge_index', 'target')

    def __post_init__(self, cap_node_full, cap_edge_full, cap_node_subg, cap_edge_subg, validate: bool):
        """
        All subgraphs sampled by the same sampler should have the same dtype, since cap_*_subg are an upper bound
        for all subgraphs under that sampler. 
        """
        if cap_node_full is not None and cap_edge_full is not None \
            and cap_node_subg is not None and cap_edge_subg is not None:
            dtype = {
                'indptr'   : np.int64,
                'indices'  : np.int64,
                'data'     : np.float32,
                'node'     : np.int64,
                'edge_index': np.int64,
                'target'   : np.int64
            }
            f_dtype = lambda n : np.uint16 if n < 2**16 else np.uint32
            if cap_node_full < 2**32:
                dtype['node'] = f_dtype(cap_node_full)
            if cap_edge_full < 2**32:
                dtype['edge_index'] = f_dtype(cap_edge_full)
            if cap_node_subg < 2**32:
                dtype['indices'] = f_dtype(cap_node_subg)
                dtype['target']  = f_dtype(cap_node_subg)
            if cap_edge_subg < 2**32:
                dtype['indptr']  = f_dtype(cap_edge_subg)
            assert set(dtype.keys()) == set(self.names_data_fields)
            for n in self.names_data_fields:
                v = getattr(self, n)
                if v is not None:
                    setattr(self, n, v.astype(dtype[n], copy=False))
            # explicitly handle data -- if it is all 1.
            if np.all(self.data == 1.):
                self.data = np.broadcast_to(np.array([1.]), self.data.size)
        if validate:
            self.check_valid()

    def _copy(self):
        datacopy = {}
        for n in self.names_data_fields:
            datacopy[n] = getattr(self, n).copy()
        return self.__class__(**datacopy)
    
    def check_valid(self):
        assert self.node.size == 0 or self.node.size == self.indptr.size - 1
        assert self.indices.size == self.data.size == self.indptr[-1]
        assert self.edge_index.size == 0 or self.edge_index.size == self.indices.size
        self.entity_enc.check_valid_(self)
        assert self.indptr.size >= 2, "Subgraph must contain at least 1 node!"
    
    @property
    def num_nodes(self):
        return self.node.size
    
    @property
    def num_edges(self):
        return self.indices.size

    @classmethod
    def cat_to_block_diagonal(cls, subgs: list):
        """ Concatenate subgraphs into a full adj matrix (i.e., into the block diagonal form) """
        offset_indices = np.cumsum([s.node.size for s in subgs])            # always int64
        offset_indptr = np.cumsum([s.edge_index.size for s in subgs])       # ^
        offset_indices[1:] = offset_indices[:-1]
        offset_indices[0] = 0
        offset_indptr[1:] = offset_indptr[:-1]
        offset_indptr[0] = 0
        node_batch = np.concatenate([s.node for s in subgs])                # keep original dtype
        edge_index_batch = np.concatenate([s.edge_index for s in subgs])    # ^
        data_batch = np.concatenate([s.data for s in subgs])                # ^
        target_batch_itr  = [s.target.astype(np.int64) for s in subgs]
        indptr_batch_itr  = [s.indptr.astype(np.int64) for s in subgs]
        indices_batch_itr = [s.indices.astype(np.int64) for s in subgs]
        target_batch, indptr_batch, indices_batch = [], [], []
        for i in range(len(subgs)):
            target_batch.append(target_batch_itr[i] + offset_indices[i])
            if i > 0:       # end of indptr1 equals beginning of indptr2. So remove one duplicate to ensure correctness. 
                indptr_batch_itr[i] = indptr_batch_itr[i][1:]
            indptr_batch.append(indptr_batch_itr[i] + offset_indptr[i])
            indices_batch.append(indices_batch_itr[i] + offset_indices[i])
        target_batch = np.concatenate(target_batch)
        indptr_batch = np.concatenate(indptr_batch)
        indices_batch = np.concatenate(indices_batch)
        entity_enc_batch = EntityEncoding.cat_batch(subgs)
        ret_subg = cls(
            indptr=indptr_batch, 
            indices=indices_batch,
            data=data_batch, 
            node=node_batch,
            edge_index=edge_index_batch,
            target=target_batch,
            entity_enc=entity_enc_batch,
            cap_node_full=2**63,        # just be safe. Note that concated subgraphs are only used for one batch. 
            cap_edge_full=2**63,
            cap_node_subg=2**63,
            cap_edge_subg=2**63,
            validate=True
        )
        return ret_subg
        
    def to_csr_sp(self):
        num_nodes = self.indptr.size - 1
        adj = sp.csr_matrix(
            (self.data, self.indices, self.indptr), shape=(num_nodes, num_nodes)
        )
        if self.indices.dtype != np.int64:
            adj.indices = adj.indices.astype(self.indices.dtype, copy=False)
            adj.indptr = adj.indptr.astype(self.indptr.dtype, copy=False)
        return adj
