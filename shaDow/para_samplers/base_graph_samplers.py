# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.sparse
from typing import Union, List
from dataclasses import dataclass, field, fields, InitVar
import scipy.sparse as sp


@dataclass
class Subgraph:
    """
    Represents the meta information of sampled subgraphs. 
    """
    # data fields
    indptr          : np.ndarray
    indices         : np.ndarray
    data            : np.ndarray
    node            : np.ndarray
    edge_index      : np.ndarray
    target          : np.ndarray
    hop             : np.ndarray
    ppr             : np.ndarray
    # init fields
    cap_node_full   : InitVar[int]=None
    cap_edge_full   : InitVar[int]=None
    cap_node_subg   : InitVar[int]=None
    cap_edge_subg   : InitVar[int]=None
    validate        : InitVar[bool]=True
    # summary
    names_data_fields = ['indptr', 'indices', 'data', 'node', 'edge_index', 'target', 'hop', 'ppr']

    def __post_init__(self, cap_node_full, cap_edge_full, cap_node_subg, cap_edge_subg, validate):
        """
        All subgraphs sampled by the same sampler should have the same dtype, since cap_*_subg are an upper bound
        for all subgraphs under that sampler. 
        """
        if cap_node_full is not None and cap_edge_full is not None \
            and cap_node_subg is not None and cap_edge_subg is not None:
            dtype = {'indptr'   : np.int64,
                     'indices'  : np.int64,
                     'data'     : np.float32,
                     'node'     : np.int64,
                     'edge_index': np.int64,
                     'target'   : np.int64,
                     'hop'      : np.int64,
                     'ppr'      : np.float32}
            f_dtype = lambda n : np.uint16 if n < 2**16 else np.uint32
            if cap_node_full < 2**32:
                dtype['node'] = f_dtype(cap_node_full)
            if cap_edge_full < 2**32:
                dtype['edge_index'] = f_dtype(cap_edge_full)
            if cap_node_subg < 2**32:
                dtype['indices'] = f_dtype(cap_node_subg)
                dtype['target']  = f_dtype(cap_node_subg)
                dtype['hop']     = f_dtype(cap_node_subg)
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
        assert self.indices.size == self.edge_index.size == self.data.size == self.indptr[-1]
        assert self.hop.size == 0 or (self.hop.size == self.indptr.size - 1)
        assert self.ppr.size == 0 or (self.ppr.size == self.indptr.size - 1)
        assert self.indptr.size >= 2, "Subgraph must contain at least 1 node!"
    
    def num_nodes(self):
        assert self.node.size == self.indptr.size - 1
        return self.node.size
    
    def num_edges(self):
        assert self.indices.size == self.edge_index.size == self.data.size == self.indptr[-1]
        return self.indices.size

    @classmethod
    def cat_to_block_diagonal(cls, subgs : list):
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
        hop_batch = np.concatenate([s.hop for s in subgs])                  # ^
        if subgs[0].ppr.size == 0:
            ppr_batch = np.array([])
        else:       # need to explicitly check due to .max() function
            ppr_batch = np.concatenate([s.ppr/s.ppr.max() for s in subgs])      # renorm ppr
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
        ret_subg = cls(
            indptr=indptr_batch, 
            indices=indices_batch,
            data=data_batch, 
            node=node_batch,
            edge_index=edge_index_batch,
            target=target_batch,
            hop=hop_batch,
            ppr=ppr_batch,
            cap_node_full=2**63,        # just be safe. Note that concated subgraphs are only used for one batch. 
            cap_edge_full=2**63,
            cap_node_subg=2**63,
            cap_edge_subg=2**63,
            validate=True
        )
        return ret_subg
        
    def to_csr_sp(self):
        num_nodes = self.indptr.size - 1
        adj = sp.csr_matrix((self.data, self.indices, self.indptr), shape=(num_nodes, num_nodes))
        if self.indices.dtype != np.int64:
            adj.indices = adj.indices.astype(self.indices.dtype, copy=False)
            adj.indptr = adj.indptr.astype(self.indptr.dtype, copy=False)
        return adj


class GraphSampler:
    """
    This is the sampler super-class. Any shallow sampler is supposed to perform
    the following meta-steps:
     1. [optional] Preprocessing: e.g., for PPR sampler, we need to calculate the
            PPR vector for each node in the training graph. This is to be performed
            only once.
            ==> Need to override the `preproc()` in sub-class
     2. Parallel sampling: launch a batch of graph samplers in parallel and sample
            subgraphs independently. For efficiency, the actual sampling operation
            happen in C++. And the classes here is mainly just a wrapper.
            ==> Need to set self.para_sampler to the appropriate C++ sampler
              in `__init__()` of the sampler sub-class
     3. Post-processing: upon getting the sampled subgraphs, we need to prepare the
            appropriate information (e.g., subgraph adj with renamed indices) to
            enable the PyTorch trainer. Also, we need to do data conversion from C++
            to Python (or, mostly numpy). Post-processing is handled via PyBind11.
    """
    def __init__(self, adj, node_target, aug_feat, args_preproc):
        """
        Inputs:
            adj             scipy sparse CSR matrix of the training graph
            node_target     1D np array storing the indices of the training nodes
            args_preproc    dict, addition arguments needed for pre-processing

        Outputs:
            None
        """
        self.adj = adj
        self.node_target = np.unique(node_target)
        self.aug_feat = aug_feat
        # size in terms of number of vertices in subgraph
        self.name_sampler = "None"
        self.node_subgraph = None
        self.preproc(**args_preproc)

    def preproc(self, **kwargs):
        raise NotImplementedError

    def par_sample(self, **kwargs):
        return self.para_sampler.par_sample()

    def helper_extract_subgraph(self, node_ids, target_ids=None):
        """
        Used for serial Python sampler (not for the parallel C++ sampler).
        Return adj of node-induced subgraph and other corresponding data struct.

        Inputs:
            node_ids        1D np array, each element is the ID in the original
                            training graph.
        Outputs:
            indptr          np array, indptr of the subg adj CSR
            indices         np array, indices of the subg adj CSR
            data            np array, data of the subg adj CSR. Since we have aggregator
                            normalization, we can simply set all data values to be 1
            subg_nodes      np array, i-th element stores the node ID of the original graph
                            for the i-th node in the subgraph. Used to index the full feats
                            and label matrices.
            subg_edge_index np array, i-th element stores the edge ID of the original graph
                            for the i-th edge in the subgraph. Used to index the full array
                            of aggregation normalization.
        """
        # Let n = num subg nodes; m = num subg edges
        node_ids = np.unique(node_ids)
        node_ids.sort()
        orig2subg = {n: i for i, n in enumerate(node_ids)}
        n = node_ids.size
        indptr = np.zeros(node_ids.size + 1)
        indices = []
        subg_edge_index = []
        subg_nodes = node_ids
        for nid in node_ids:
            idx_s, idx_e = self.adj.indptr[nid], self.adj.indptr[nid + 1]
            neighs = self.adj.indices[idx_s : idx_e]
            for i_n, n in enumerate(neighs):
                if n in orig2subg:
                    indices.append(orig2subg[n])
                    indptr[orig2subg[nid] + 1] += 1
                    subg_edge_index.append(idx_s + i_n)
        indptr = indptr.cumsum().astype(np.int64)
        indices = np.array(indices)
        subg_edge_index = np.array(subg_edge_index)
        data = np.ones(indices.size)
        assert indptr[-1] == indices.size == subg_edge_index.size
        if target_ids is not None:
            return indptr, indices, data, subg_nodes, subg_edge_index,\
                np.array([orig2subg[t] for t in target_ids])
        else:
            return indptr, indices, data, subg_nodes, subg_edge_index


class NodeIIDBase(GraphSampler):
    def __init__(self, adj, node_target, aug_feat):
        self.name = 'nodeIID'
        super().__init__(adj, node_target, aug_feat, {})

    def preproc(self, **kwargs):
        pass



class KHopSamplingBase(GraphSampler):
    """
    The sampler performs k-hop sampling, by following the steps:
     1. Randomly pick `size_root` number of root nodes from all training nodes;
     2. Sample hop-`k` neighborhood from the roots. A node at hop-i will fanout to
        at most `budget` nodes at hop-(i+1)
     3. Generate node-induced subgraph from the nodes touched by the random walk.
    If budget == -1, then we will expand all hop-(i+1) neighbors without any subsampling
    """
    def __init__(self, adj, node_target, aug_feat, size_root, depth, budget):
        """
        Inputs:
            adj             see super-class
            node_target     see super-class
            size_root       int, number of root nodes randomly picked
            depth           int, number of hops to expand
            budget          int, number of hop-(i+1) neighbors to expand

        Outputs:
            None
        """
        self.size_root = size_root
        self.depth = depth
        self.budget = budget
        self.name = "khop"
        super().__init__(adj, node_target, aug_feat, {})

    def preproc(self, **kwargs):
        pass


class PPRSamplingBase(GraphSampler):
    """
    The sampler performs sampling based on PPR score
    """
    def __init__(self, adj, node_target, aug_feat, size_root, k, alpha=0.85, epsilon=1e-5, threshold=0):
        """
        Inputs:
            adj             see super-class
            node_target     see super-class
            size_root       int, number of root nodes randomly picked
            k               int, number of hops to expand
            budget          int, number of hop-(i+1) neighbors to expand

        Outputs:
            None
        """
        self.size_root = size_root
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.threshold = threshold
        self.name = "ppr"
        super().__init__(adj, node_target, aug_feat, {})

    def preproc(self, **kwargs):
        raise NotImplementedError

