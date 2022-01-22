# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
inheritance from base_graph_samplers: python version without any backend
"""

from graph_engine.frontend.samplers_base import GraphSampler
import numpy as np
import torch
import torch.nn.functional as F
import random
import torch.multiprocessing as mp

# the simplest sampler
class NodeSamplingVanillaPython(GraphSampler):
    """
    A basic sampler: just pick nodes uniformly at random and return the
    node-induced subgraph.
    """
    def __init__(self, adj_train, node_train, size_subgraph, num_subg_per_batch=200):
        super().__init__(adj_train, node_train, size_subgraph, {}, num_subg_per_batch=num_subg_per_batch)
        self.backend = 'python'

    def parallel_sample(self, **kwargs):
        node_ids = np.random.choice(self.node_train, self.size_subgraph)
        ret = self.helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret

    def preproc(self):
        pass


class KHopSamplingPy(GraphSampler):
    def __init__(self, adj, size_subgraph, size_root, depth, budget,
            num_cpu_core=1, fix_target=False, num_subg_per_batch=200, **kwargs):
        super().__init__(adj, size_subgraph, {}, num_subg_per_batch=num_subg_per_batch)
        self.depth = depth
        self.budget = budget
        self.size_root = size_root
        self.fix_target = fix_target
        self.backend = 'python'
        self.name = 'khop'

    def _parallel_sample(self, roots_batch, **kwargs):
        ret = []
        for i in range(0, roots_batch.size, self.size_root):
            root_ids = roots_batch[i : i + self.size_root]
            node_lvl = []
            node_lvl.append(root_ids)
            for _i in range(self.depth):
                node_cur = []
                for n in node_lvl[-1]:
                    _idx_start = self.adj.indptr[n]
                    _idx_end = self.adj.indptr[n + 1]
                    if _idx_start == _idx_end:
                        continue
                    candidates = self.adj.indices[_idx_start : _idx_end]
                    selection = candidates if self.budget == -1 else np.random.choice(candidates, self.budget)
                    node_cur.append(np.unique(selection))
                node_cur = np.concatenate(node_cur)
                node_lvl.append(np.unique(node_cur))
            node_ids = np.unique(np.concatenate(node_lvl))
            target_ids = root_ids if self.fix_target else None
            ret.append(self.helper_extract_subgraph(node_ids, target_ids))
        return ret

    def preproc(self):
        pass
