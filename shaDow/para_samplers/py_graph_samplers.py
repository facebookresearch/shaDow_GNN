# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from base_graph_samplers import GraphSampler
import numpy as np


# the simplest sampler
class NodeSamplingVanillaPython(GraphSampler):
    """
    A basic sampler: just pick nodes uniformly at random and return the
    node-induced subgraph.
    """
    def __init__(self, adj_train, node_train, size_subgraph):
        super().__init__(adj_train, node_train, size_subgraph, {})

    def par_sample(self, **kwargs):
        node_ids = np.random.choice(self.node_train, self.size_subgraph)
        ret = self.helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret

    def preproc(self):
        pass


# --------------------------------------------------
# BELOW: experiments for "Deep GNN, Shallow Sampler"
# --------------------------------------------------


class KHopVanillaPython(GraphSampler):
    def __init__(self, adj, node_pool, size_subgraph, size_roots, k, budget,
            num_cpu_core=1, fix_target=False, **kwargs):
        super().__init__(adj, node_pool, size_subgraph, {})
        self.k = k
        self.budget = budget
        self.size_roots = size_roots
        self.fix_target = fix_target

    def par_sample(self, **kwargs):
        root_ids = np.unique(np.random.choice(self.node_train, self.size_roots))
        node_lvl = []
        node_lvl.append(root_ids)
        for _i in range(self.k):
            node_cur = []
            for n in node_lvl[-1]:
                _idx_start = self.adj_train.indptr[n]
                _idx_end = self.adj_train.indptr[n + 1]
                if _idx_start == _idx_end:
                    continue
                candidates = self.adj_train.indices[_idx_start : _idx_end]
                if self.budget == -1:
                    selection = candidates
                else:
                    selection = np.random.choice(candidates, self.budget)
                node_cur.append(np.unique(selection))
            node_cur = np.concatenate(node_cur)
            node_lvl.append(np.unique(node_cur))
        node_ids = np.unique(np.concatenate(node_lvl))
        if self.fix_target:
            target_ids = root_ids
        else:
            target_ids = None
        ret = self.helper_extract_subgraph(node_ids, target_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret

    def preproc(self):
        pass
