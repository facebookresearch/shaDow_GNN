# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

from typing import Union, List
from collections import deque
from copy import deepcopy
import numpy as np
import scipy.sparse as sp
import torch
from shaDow import TRAIN, VALID, TEST
from shaDow.globals import REUSABLE_SAMPLER, device
import shaDow.para_samplers.cpp_graph_samplers as gsp
from shaDow.para_samplers.base_graph_samplers import Subgraph
from collections import defaultdict
from dataclasses import dataclass, field, fields, InitVar

import time



@dataclass
class PoolSubg:
    """
    Collection of sampled subgraphs returned from the C++ sampler.
    """
    data            : List[deque] = None
    # book-keeper
    num_subg        : int = 0           # num subgraphs should be identical across different branches of ensemble
    num_ensemble    : int = 0

    def __post_init__(self):
        self.data = [deque() for i in range(self.num_ensemble)]
        self.num_subg = 0

    def add(self, i_ens : int, subgs : list):
        for s in subgs:
            assert type(s) == Subgraph
            self.data[i_ens].append(s)
        assert self.num_subg <= len(self.data[i_ens])
        self.num_subg = len(self.data[i_ens])

    def collate(self, batch_size):
        """
        Concatenate batch_size number of subgraphs in the pool, into a single adj matrix (block diagonal form)
        e.g., ensemble = 1, batch_size = 3, and the 3 subgraphs are of size 2, 1, 3. Then the output will be:
        * subg_cat:
            x x 0 0 0 0
            x x 0 0 0 0
            0 0 x 0 0 0
            0 0 0 x x x
            0 0 0 x x x
            0 0 0 x x x
        * size_cat:
            [2, 1, 3]
        """
        subg_cat, size_cat = [], []
        for i in range(self.num_ensemble):
            subg_to_cat = []
            for _ in range(batch_size):
                subg_to_cat.append(self.data[i].popleft())
            subg_cat.append(Subgraph.cat_to_block_diagonal(subg_to_cat))
            size_cat.append([sc.indptr.size - 1 for sc in subg_to_cat])
            assert sum(size_cat[-1]) == subg_cat[-1].indptr.size - 1
        self.num_subg -= batch_size
        return subg_cat, size_cat


@dataclass
class CacheSubg:
    """
    Caching the previously sampled subgraph to be reused by later on training epochs
    """
    data            : List[dict] = None
    # book-keeper
    num_recorded    : List[int] = None
    # init var
    _num_ens        : InitVar[int] = 0
    def __post_init__(self, _num_ens : int):
        self.data = [{} for _ in range(_num_ens)]
        self.num_recorded = [0 for _ in range(_num_ens)]
    
    def get(self, i_ens : int, i_subg : int):
        return self.data[i_ens][i_subg]

    def set(self, i_ens : int, i_subg : int, subg : Subgraph):
        self.data[i_ens][i_subg] = subg
        self.num_recorded[i_ens] += 1

    def is_empty(self, i_ens : int):
        return len(self.data[i_ens]) == 0



class MinibatchShallowSampler:
    FULL = 0
    SUBG = 1
    def __init__(self, 
                name_data, 
                dir_data, 
                adjs, 
                node_set, 
                sampler_config_ensemble, 
                aug_feats,
                train_params, 
                feat_full, 
                label_full, 
                is_transductive : bool,
                parallelism : int, 
                full_tensor_on_gpu : bool=True,
                bin_adj_files=None,
                nocache_modes : set={*()},
                optm_level='high'
            ):
        self.optm_level = optm_level
        self.aug_feats = aug_feats
        self.dev_torch = device
        self.dir_data, self.name_data = dir_data, name_data
        self.batch_num, self.batch_size = -1, {TRAIN: 0, VALID: 0, TEST: 0}
        self.node_set = node_set
        self.is_transductive = is_transductive
        assert set(adjs.keys()) == set([TRAIN, VALID, TEST])
        self.adj = adjs

        self.feat_full, self.label_full = feat_full, label_full
        assert type(label_full) == torch.Tensor and type(feat_full) == torch.Tensor
        if full_tensor_on_gpu:
            self.feat_full = self.feat_full.to(self.dev_torch)
            self.label_full = self.label_full.to(self.dev_torch)

        self.num_train_nodes = self.node_set[TRAIN].size

        self.idx_target_evaluated = {VALID: 0, TEST: 0, TRAIN: 0}       # for keeping track if evaluation minibatches has sweeped the whole val/test sets
        self.end_epoch = {VALID: False, TEST: False, TRAIN: False}
        self.nocache_modes = nocache_modes
        self.graph_sampler = {TRAIN: None, VALID: None, TEST: None}
        sampler_config_ensemble = deepcopy(sampler_config_ensemble)
        self.num_ensemble = 0
        for sc in sampler_config_ensemble['configs']:
            num_ens_cur = [len(v) for k, v in sc.items() if k != 'method']
            if len(num_ens_cur) == 0:
                self.num_ensemble += 1
            else:
                assert max(num_ens_cur) == min(num_ens_cur)
                self.num_ensemble += num_ens_cur[0]
        if "full" in [c['method'] for c in sampler_config_ensemble['configs']]:
            # treat FULL sampler as no sampling. Also no ensemble under FULL sampler
            assert self.num_ensemble == 1
            for m in [TRAIN, VALID, TEST]:
                self.batch_size[m] = self.node_set[m].size
            self.mode_sample = self.FULL
        else:
            self.record_subgraphs = {}
            self.args_sampler_init = [sampler_config_ensemble, parallelism, bin_adj_files]
            if self.optm_level != 'high':
                self.instantiate_sampler(*self.args_sampler_init)
                assert len(self.graph_sampler[TRAIN].sampler_list) \
                    == len(self.graph_sampler[VALID].sampler_list) \
                    == len(self.graph_sampler[TEST].sampler_list)
                for m in [TRAIN, VALID, TEST]:
                    if m not in self.nocache_modes:
                        self.record_subgraphs[m] = ["record" if g.name in REUSABLE_SAMPLER else "none" for g in self.graph_sampler[m].sampler_list]
                    else:
                        self.record_subgraphs[m] = ['noncache'] * self.num_ensemble
            # -------- storing the subgraph samples -------
            # only for deterministic samplers: e.g., PPR
            self.cache_subg, self.pool_subg = {}, {}
            for m in [TRAIN, VALID, TEST]:
                self.cache_subg[m] = CacheSubg(_num_ens=self.num_ensemble)
                self.pool_subg[m] = PoolSubg(num_ensemble=self.num_ensemble)
            self.mode_sample = self.SUBG

        # --------- LOGGING ---------
        self.max_hop_to_profile = 5
        self.max_ppr_to_profile = 1
        

    def _get_cur_batch_size(self, mode):
        self.end_epoch[mode] = False
        return min(self.node_set[mode].size - self.idx_target_evaluated[mode], self.batch_size[mode])

    def _update_batch_stat(self, mode, batch_size):
        self.batch_num += 1
        self.idx_target_evaluated[mode] += batch_size
        # TODO: may move this to epoch_start reset. --> Then you don't need end_epoch. Just check from idx_target_evaluated
        if self.idx_target_evaluated[mode] >= self.node_set[mode].size:
            assert self.idx_target_evaluated[mode] == self.node_set[mode].size
            self.idx_target_evaluated[mode] = 0
            self.end_epoch[mode] = True
            if self.graph_sampler[mode] is not None:
                assert self.pool_subg[mode].num_subg == 0
                assert all(len(d) == 0 for d in self.pool_subg[mode].data)

    def shuffle_nodes(self, mode):
        if self.graph_sampler[mode] is not None:
            self.graph_sampler[mode].shuffle_targets()

    
    def epoch_start_reset(self, epoch, mode):
        """
        Reset structs related with later on reuse of sampled subgraphs. 
        """
        self.batch_num = -1
        if self.graph_sampler[mode] is None and self.mode_sample == self.SUBG:
            self.instantiate_sampler(*self.args_sampler_init, modes=[mode])
            if mode not in self.nocache_modes:
                self.record_subgraphs[mode] = ["record" if g.name in REUSABLE_SAMPLER else "none" for g in self.graph_sampler[mode].sampler_list]
            else:
                self.record_subgraphs[mode] = ['noncache'] * self.num_ensemble
        elif self.mode_sample == self.FULL:
            return    
        self.graph_sampler[mode].return_target_only = []
        for i in range(len(self.record_subgraphs[mode])):
            if self.record_subgraphs[mode][i] == "reuse":
                self.graph_sampler[mode].return_target_only.append(True)
            else:
                self.graph_sampler[mode].return_target_only.append(False)


    def is_end_epoch(self, mode):
        return self.end_epoch[mode]

    def epoch_end_reset(self, mode):
        self.end_epoch[mode] = False
        if self.mode_sample == self.FULL:
            return
        for i in range(len(self.record_subgraphs[mode])):
            if self.record_subgraphs[mode][i] == "record" and not self.cache_subg[mode].is_empty(i):
                self.record_subgraphs[mode][i] = "reuse"
        if self.optm_level == 'high' and self.mode_sample != self.FULL and all(r == 'reuse' for r in self.record_subgraphs[mode]):
            self.graph_sampler[mode].drop_full_graph_info()
                  

    def instantiate_sampler(self, sampler_config_ensemble, parallelism, bin_adj_files, modes=[TRAIN, VALID, TEST]):
        sampler_config_ensemble_ = deepcopy(sampler_config_ensemble)
        config_ensemble = []
        # e.g., input: [{"method": "ppr", "k": [50, 10]}, {"method": "khop", "depth": [2], "budget": [10]}]
        #       output: [{"method": "ppr", "k": 50}, {"method": "ppr", "k": 10}, {"method": "khop", "depth": 2, "budget": 10}]
        for cfg in sampler_config_ensemble_["configs"]:  # different TYPEs of samplers
            method = cfg.pop('method')
            cnt_cur_sampler = [len(v) for k, v in cfg.items()]
            assert len(cnt_cur_sampler) == 0 or max(cnt_cur_sampler) == min(cnt_cur_sampler)
            cnt_cur_sampler = 1 if len(cnt_cur_sampler) == 0 else cnt_cur_sampler[0]
            cfg['method'] = [method] * cnt_cur_sampler
            cfg_decoupled = [{k: v[i] for k, v in cfg.items()} for i in range(cnt_cur_sampler)]
            config_ensemble.extend(cfg_decoupled)
        self.num_ensemble = len(config_ensemble)
        config_ensemble_mode = {}
        for m in modes:
            self.batch_size[m] = sampler_config_ensemble_["batch_size"]
            config_ensemble_mode[m] = deepcopy(config_ensemble)
        for cfg in config_ensemble:
            assert "method" in cfg
            assert "size_root" not in cfg or cfg["size_root"] == 1
        for cfg_mode, cfg_ensemble in config_ensemble_mode.items():
            for cfg in cfg_ensemble:
                cfg["size_root"] = 1                 # we want each target to have its own subgraph
                cfg["fix_target"] = True             # i.e., we differentiate root node from the neighbor nodes (compare with GraphSAINT)
                cfg["sequential_traversal"] = True   # (mode != "train")
                if cfg["method"] == "ppr":
                    cfg["type_"] = cfg_mode
                    cfg['name_data'] = self.name_data
                    cfg["dir_data"] = self.dir_data
                    cfg['is_transductive'] = self.is_transductive
            aug_feat_ens = [self.aug_feats] * len(cfg_ensemble)
            self.graph_sampler[cfg_mode] = gsp.GraphSamplerEnsemble(
                    self.adj[cfg_mode], self.node_set[cfg_mode], cfg_ensemble, aug_feat_ens,
                    max_num_threads=parallelism, num_subg_per_batch=200, bin_adj_files=bin_adj_files[cfg_mode]
                )

    def par_graph_sample(self, mode):
        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        """
        subg_ens_l_raw = self.graph_sampler[mode].par_sample_ensemble()
        for i, subg_l_raw in enumerate(subg_ens_l_raw):
            subg_ens_l = None
            if self.record_subgraphs[mode][i] == "record":
                for subg in subg_l_raw:
                    assert subg.target.size == 1
                    id_root = subg.node[subg.target][0]
                    self.cache_subg[mode].set(i, id_root, subg)
                subg_ens_l = subg_l_raw
            elif self.record_subgraphs[mode][i] == "reuse":
                subg_ens_l = []
                for subg in subg_l_raw:
                    assert subg.node.size == 1
                    id_root = subg.node[0]
                    subg_ens_l.append(self.cache_subg[mode].get(i, id_root))
            elif self.record_subgraphs[mode][i] in ['noncache', 'none']:
                subg_ens_l = subg_l_raw
            else:
                raise NotImplementedError
            self.pool_subg[mode].add(i, subg_ens_l)

    def one_batch(self, mode=TRAIN, ret_raw_idx=False):
        """
        Prepare one batch of training subgraph. For each root, the sampler returns the subgraph adj separatedly. 
        To improve the computation efficiency, we concatenate the batch number of separate adj into a single big adj. 
        i.e., the resulting adj is of the block-diagnol form. 
        """
        if self.graph_sampler[mode] is None:    # no sampling, return the full graph. 
            self._update_batch_stat(mode, self.batch_size[mode])
            targets = self.node_set[mode]
            assert ret_raw_idx, "None subg mode should only be used in preproc!"
            return {"adj_ens": [self.adj[mode]], "feat_ens": [self.feat_full], "label_ens": [self.label_full],
                    "size_subg_ens": None, "target_ens": [targets], "feat_aug_ens": None, "idx_raw": [np.arange(self.adj[mode].shape[0])]}
        t0 = time.time()
        batch_size_ = self._get_cur_batch_size(mode)
        while self.pool_subg[mode].num_subg < batch_size_:
            self.par_graph_sample(mode)
        if batch_size_ != self.batch_size[mode]:
            assert batch_size_ < self.batch_size[mode] and self.pool_subg[mode].num_subg == batch_size_
            assert self.graph_sampler[mode].para_sampler.get_idx_root() == 0
            assert self.graph_sampler[mode].para_sampler.is_seq_root_traversal()
        adj_ens, feat_ens, target_ens, label_ens, feat_aug_ens = [], [], [], [], []
        label_idx = None
        subgs_ens, size_subg_ens = self.pool_subg[mode].collate(batch_size_)
        size_subg_ens = torch.tensor(size_subg_ens).to(self.dev_torch)
        feat_aug_ens = []
        for subgs in subgs_ens:       
            assert subgs.target.size == batch_size_
            if label_idx is None:
                label_idx = subgs.node[subgs.target]
            else:
                assert np.all(label_idx == subgs.node[subgs.target])
            adj_ens.append(subgs.to_csr_sp())
            feat_ens.append(self.feat_full[subgs.node].to(self.dev_torch))
            target_ens.append(subgs.target)
            label_ens.append(self.label_full[subgs.node].to(self.dev_torch))
            feat_aug_ens.append({})
            if 'hops' in self.aug_feats:
                feat_aug_ens[-1]['hops'] = torch.tensor(self.hop2onehot_vec(subgs.hop).astype(np.float32)).to(self.dev_torch)
            if 'pprs' in self.aug_feats:
                feat_aug_ens[-1]['pprs'] = torch.tensor(self.ppr2onehot_vec(subgs.ppr).astype(np.float32)).to(self.dev_torch)
        self._update_batch_stat(mode, batch_size_)
        ret = {"adj_ens"        : adj_ens, 
               "feat_ens"       : feat_ens, 
               "label_ens"      : label_ens, 
               "size_subg_ens"  : size_subg_ens, 
               "target_ens"     : target_ens, 
               "feat_aug_ens"   : feat_aug_ens}
        if ret_raw_idx:
            assert len(adj_ens) == 1
            ret["idx_raw"] = [subgs.node for subgs in subgs_ens]
        return ret

    def disable_cache(self, mode):
        if self.mode_sample is not self.FULL:
            self.record_subgraphs[mode] = ['noncache'] * self.num_ensemble
            self.nocache_modes.add(mode)

    def get_aug_dim(self, aug_type):
        if aug_type == "hops":
            return self.max_hop_to_profile + 2
        elif aug_type == "pprs":
            return self.max_ppr_to_profile
        else:
            raise NotImplementedError

    def hop2onehot_vec(self, hops_raw):
        assert len(hops_raw.shape) == 1
        ret = np.zeros((hops_raw.size, self.max_hop_to_profile + 2))
        valid_h = [-1, 0] + [i for i in range(1, self.max_hop_to_profile + 1)]
        for i in valid_h:
            ret[np.where(hops_raw == i)[0], i + 1] = 1
        return ret

    def ppr2onehot_vec(self, pprs_raw):
        assert len(pprs_raw.shape) == 1
        ret = np.zeros((pprs_raw.size, self.max_ppr_to_profile))
        cond_filter = [0.25**i for i in range(self.max_ppr_to_profile)]
        cond_filter += [0]
        for i in range(self.max_ppr_to_profile):
            ret[np.where(np.logical_and(pprs_raw <= cond_filter[i], pprs_raw >= cond_filter[i+1])), i] = 1
        return ret
