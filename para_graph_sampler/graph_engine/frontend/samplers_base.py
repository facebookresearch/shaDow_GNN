import numpy as np
from typing import Union, List
import numbers
from graph_engine.frontend.graph import EntityEncoding, Subgraph


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
    def __init__(self, adj, aug_feat, args_preproc, num_subg_per_batch=200):
        """
        Inputs:
            adj             scipy sparse CSR matrix of the training graph
            args_preproc    dict, addition arguments needed for pre-processing

        Outputs:
            None
        """
        self.adj = adj
        self.node_target = None
        self.aug_feat = aug_feat
        # size in terms of number of vertices in subgraph
        self.name_sampler = "None"
        self.node_subgraph = None
        self.preproc(**args_preproc)
        self.backend = None
        self.num_subg_per_batch = num_subg_per_batch
        self.idx_root_traversed = 0     # use this idx to sequentially traverse the node_target. 

    @property
    def num_nodes(self):
        return self.adj.indptr.size - 1
    
    @property
    def num_edges(self):
        return self.adj.indices.size

    def preproc(self, **kwargs):
        raise NotImplementedError

    def parallel_sample(self, **kwargs):
        """
        This is a wrapper of _parallel_sample
        """
        roots_batch = self.pre_batch_select_roots()
        if 'return_target_only' in kwargs and kwargs['return_target_only']:
            ret = []
            arr_dummy = np.array([])
            for i in range(0, roots_batch.size, self.size_root):
                args = [arr_dummy] * 3 + [roots_batch[i : i + self.size_root]] + [arr_dummy] * 5
                ret.append(Subgraph(*args, validate=False))
        else:
            ret = self._parallel_sample(roots_batch, **kwargs)
        self.post_batch_update_idx()
        return ret

    def _parallel_sample(self, **kwargs):
        raise NotImplementedError
    
    def post_batch_update_idx(self):
        self.idx_root_traversed += self.num_subg_per_batch
        if self.idx_root_traversed >= self.node_target.size:
            self.idx_root_traversed = 0     # reset for next epoch
    
    def pre_batch_select_roots(self):
        return self.node_target[self.idx_root_traversed : self.idx_root_traversed + self.num_subg_per_batch]

    def shuffle_targets(self, targets_shuffled=None):
        if targets_shuffled is not None and targets_shuffled.size > 0:
            self.node_target = targets_shuffled
        else:
            self.node_target = np.random.permutation(self.node_target)

    def helper_extract_subgraph(
        self, 
        node_ids, 
        target_ids=None, 
        type_aug={'hop'}, 
        cap_node_subg=None, 
        cap_edge_subg=None, 
        remove_target_links=True
    ):
        """ NOTE this func is not used by exp in our paper. Just provided for your reference. 
        Used for serial Python sampler (not for the parallel C++ sampler).
        Return adj of node-induced subgraph and other corresponding data struct.

        Inputs:
            node_ids        1D np array, each element is the ID in the original
                            training graph.
            type_aug        types of feature augmentation. currently only support hop
            remove_target_links     if true, will not include the edges among target nodes
        Outputs: --> Subgraph
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
        if target_ids is None:
            target_set = {}
            target = np.array([])
        elif type(target_ids) == np.ndarray:
            target_set = set(target_ids)
            target = np.array([orig2subg[t] for t in target_ids])
        elif isinstance(target_ids, numbers.Number):
            target_set = {target_ids}
            target = np.array([orig2subg[target_ids]])
        else:
            raise NotImplementedError
        for nid in node_ids:
            idx_s, idx_e = self.adj.indptr[nid], self.adj.indptr[nid + 1]
            neighs = self.adj.indices[idx_s : idx_e]
            for i_n, n in enumerate(neighs):
                if n in orig2subg and (not remove_target_links or n == nid or
                        n not in target_set or nid not in target_set):
                    indices.append(orig2subg[n])
                    indptr[orig2subg[nid] + 1] += 1
                    subg_edge_index.append(idx_s + i_n)
        indptr = indptr.cumsum().astype(np.int64)
        indices = np.array(indices)
        subg_edge_index = np.array(subg_edge_index)
        data = np.ones(indices.size)
        assert indptr[-1] == indices.size == subg_edge_index.size
        if cap_node_subg is None or cap_node_subg < 1:
            cap_node_subg = self.adj.indptr.size - 1
        cap_node_subg = min(cap_node_subg, self.adj.indptr.size - 1)
        if cap_edge_subg is None or cap_edge_subg < 1:
            cap_edge_subg = self.adj.indices.size
        cap_edge_subg = min(cap_edge_subg, self.adj.indices.size)
        ret_subg_enc = EntityEncoding(np.array([]), np.array([]), np.array([]))
        ret_subg = Subgraph(
            indptr, 
            indices, 
            data, 
            subg_nodes, 
            subg_edge_index, 
            target, 
            ret_subg_enc,
            cap_node_full=self.adj.indptr.size - 1, 
            cap_edge_full=self.adj.indices.size,
            cap_node_subg=cap_node_subg, 
            cap_edge_subg=cap_edge_subg, 
            validate=True
        )
        if type_aug is None:
            type_aug = {}
        if 'hop' in type_aug:
            assert target.size == 1, 'only node classification task can be augmented with hops'
            ret_subg.entity_enc.fill_hops()
        if 'drnl' in type_aug:
            assert target.size == 2, 'only link prediction task can be augmented with drnl'
            ret_subg.entity_enc.fill_drnl()
        return ret_subg


class NodeIIDBase(GraphSampler):
    def __init__(self, adj, aug_feat, num_subg_per_batch=200):
        self.name = 'nodeIID'
        super().__init__(adj, aug_feat, {}, num_subg_per_batch=num_subg_per_batch)

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
    def __init__(self, adj, aug_feat, size_root, depth, budget, num_subg_per_batch=200):
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
        super().__init__(adj, aug_feat, {}, num_subg_per_batch=num_subg_per_batch)

    def preproc(self, **kwargs):
        pass


class PPRSamplingBase(GraphSampler):
    """
    The sampler performs sampling based on PPR score
    """
    def __init__(
        self, 
        adj, 
        aug_feat, 
        size_root, 
        k, 
        alpha=0.85, 
        epsilon=1e-5, 
        threshold=0, 
        num_subg_per_batch=200, 
        args_preproc={}
    ):
        """
        Inputs:
            adj             see super-class
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
        super().__init__(adj, aug_feat, args_preproc, num_subg_per_batch=num_subg_per_batch)

    def preproc(self, **kwargs):
        raise NotImplementedError

