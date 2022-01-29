# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Optional, List, Dict, Any
from collections import deque
from copy import deepcopy
import numpy as np
import scipy.sparse as sp
import torch
from graph_engine.frontend import TRAIN, VALID, TEST, STR2MODE
from shaDow.globals import REUSABLE_SAMPLER, device
from graph_engine.frontend.samplers_ensemble import GraphSamplerEnsemble
from graph_engine.frontend.samplers_base import Subgraph
from shaDow.profiler import SubgraphProfiler
from dataclasses import dataclass, InitVar
from torch_geometric.utils import negative_sampling, add_self_loops, to_undirected



@dataclass
class PoolSubgraph:
    """
    Collection of sampled subgraphs returned from the C++ sampler.
    """
    data: List[deque] = None
    # book-keeper
    num_subg: int = 0           # num subgraphs should be identical across different branches of ensemble
    num_ensemble: int = 0

    def __post_init__(self):
        self.data = [deque() for i in range(self.num_ensemble)]
        self.num_subg = 0

    def add(self, i_ens: int, subgs: list):
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
        NOTE works for both node and link tasks, where there can be 1 or 2 targets per subgraph.
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
class CachedSubgraph:
    """
    Caching the previously sampled subgraph to be reused by subsequent training epochs
    """
    data: List[dict] = None
    # book-keeper
    num_recorded: List[int] = None
    # init var
    _num_ens: InitVar[int] = 0
    def __post_init__(self, _num_ens: int):
        self.data = [{} for _ in range(_num_ens)]
        self.num_recorded = [0 for _ in range(_num_ens)]
    
    def get(self, i_ens: int, i_subg: int):
        return self.data[i_ens][i_subg]

    def set(self, i_ens: int, i_subg: int, subg: Subgraph):
        self.data[i_ens][i_subg] = subg
        self.num_recorded[i_ens] += 1

    def is_empty(self, i_ens : int):
        return len(self.data[i_ens]) == 0


@dataclass
class OneBatchSubgraph:
    """
    data returned by one minibatch of MinibatchShallowExtractor.
    Support subgraph ensemble
    """
    adj_ens: List[sp.csr.csr_matrix]
    feat_ens: List[Union[np.ndarray, torch.tensor]]
    label: Union[np.ndarray, torch.tensor]
    size_subg_ens: Union[np.ndarray, torch.tensor, None]
    target_ens: List[Union[np.ndarray, torch.tensor]]
    feat_aug_ens: Optional[List[Dict[str, Any]]]
    idx_raw: List[Union[np.ndarray, torch.tensor, None]]=None
    
    @property
    def num_ens(self):
        return len(self.adj_ens)

    @property
    def batch_size(self):
        return self.target_ens[0].size
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        assert len(self.adj_ens) == self.num_ens
        assert len(self.feat_ens) == self.num_ens
        if self.size_subg_ens is not None:
            assert self.size_subg_ens.shape[0] == self.num_ens
        assert len(self.target_ens) == self.num_ens
        if self.feat_aug_ens is not None:
            assert len(self.feat_aug_ens) == self.num_ens
        if self.idx_raw is not None:
            assert len(self.idx_raw) == self.num_ens
    
    def pop_idx_raw(self):
        ret = self.idx_raw
        self.idx_raw = None
        return ret
    
    def to_dict(self, keys=None):
        if keys is None:
            keys = self.__dataclass_fields__
        return {
            k: getattr(self, k) for k in keys
        }


class MinibatchShallowExtractor:
    """
    NOTE on batch_size:
    
    For node classification task, batch size means the number of target nodes per batch / gradient update
    For link prediction task, batch size means the number of target edges per batch / gradient update

    i.e., for link prediction, number of 'target nodes' = 2 x batch_size
    """
    FULL = 0
    SUBG = 1
    def __init__(
        self, 
        name_data, 
        dir_data, 
        adjs, 
        entity_set, 
        sampler_config_ensemble, 
        aug_feats,
        percent_per_epoch,      # downsample nodes / edges per epoch
        feat_full, 
        label_full, 
        dim_feat_raw : int,
        is_transductive : bool,
        parallelism : int, 
        full_tensor_on_gpu : bool=True,
        bin_adj_files=None,
        nocache_modes : set={*()},
        optm_level='high',
        seed_cpp=-1,
        metrics_profile={'running': {}, 'global': {'hops', 'sizes'}},
    ):
        self.optm_level = optm_level
        self.aug_feats = aug_feats
        self.dev_torch = device
        self.dir_data, self.name_data = dir_data, name_data
        self.batch_num = -1
        self.batch_size = {TRAIN: 0, VALID: 0, TEST: 0}
        self.raw_entity_set = entity_set
        assert set([TRAIN, VALID, TEST]).issubset(set(self.raw_entity_set.keys()))
        if type(self.raw_entity_set[TRAIN]) == dict:
            for _k, _v in self.raw_entity_set.items():
                assert set(_v.keys()).issubset(set(['pos', 'neg']))
            assert label_full is None
            self.prediction_task = 'link'
        else:
            self.prediction_task = 'node'
        self.entity_epoch = {m: None for m in [TRAIN, VALID, TEST]}
        self.label_epoch = {m: None for m in [TRAIN, VALID, TEST]}      # to be set each epoch
        self.is_transductive = is_transductive
        assert set(adjs.keys()) == set([TRAIN, VALID, TEST])
        self.adj = adjs

        self.feat_full, self.label_full = feat_full, label_full
        self.dim_feat_raw = dim_feat_raw
        assert (label_full is None or type(label_full) == torch.Tensor) and type(feat_full) == torch.Tensor
        if full_tensor_on_gpu:
            self.feat_full = self.feat_full.to(self.dev_torch)
            if self.label_full is not None:
                self.label_full = self.label_full.to(self.dev_torch)
        
        # for keeping track if evaluation minibatches has sweeped the whole val/test sets
        self.idx_entity_evaluated = {VALID: 0, TEST: 0, TRAIN: 0}
        self.end_epoch = {VALID: False, TEST: False, TRAIN: False}
        if percent_per_epoch is None:
            self.percent_per_epoch = {TRAIN: 1., VALID: 1., TEST: 1.}
        else:
            self.percent_per_epoch = {}
            for k, v in percent_per_epoch.items():
                self.percent_per_epoch[STR2MODE[k]] = float(v)
        self.nocache_modes = nocache_modes if self.prediction_task == 'node' else {TRAIN, VALID, TEST}
        self.graph_sampler = {TRAIN: None, VALID: None, TEST: None}
        sampler_config_ensemble = deepcopy(sampler_config_ensemble)
        self.num_ensemble = 0
        self.seed_cpp = seed_cpp
        for sc in sampler_config_ensemble['configs']:
            num_ens_cur = [len(v) for k, v in sc.items() if k != 'method']
            if len(num_ens_cur) == 0:
                self.num_ensemble += 1
            else:
                assert max(num_ens_cur) == min(num_ens_cur)
                self.num_ensemble += num_ens_cur[0]
        self.is_stochastic_sampler = {}
        if "full" in [c['method'] for c in sampler_config_ensemble['configs']]:
            # treat FULL sampler as no sampling. Also no ensemble under FULL sampler
            assert self.prediction_task == 'node' and self.num_ensemble == 1
            for m in [TRAIN, VALID, TEST]:
                self.batch_size[m] = self.raw_entity_set[m].shape[0]
                self.is_stochastic_sampler[m] = False
            self.mode_sample = self.FULL
        else:
            self.record_subgraphs = {}
            self.args_sampler_init = [sampler_config_ensemble, parallelism, bin_adj_files]
            # -------- storing the subgraph samples -------
            # only for deterministic samplers: e.g., PPR
            self.cache_subg, self.pool_subg = {}, {}
            for m in [TRAIN, VALID, TEST]:
                self.cache_subg[m] = CachedSubgraph(_num_ens=self.num_ensemble)
                self.pool_subg[m] = PoolSubgraph(num_ensemble=self.num_ensemble)
            self.mode_sample = self.SUBG

        self.dtype = torch.get_default_dtype()
        # --------- LOGGING ---------
        self.dim_1hot_hop: int = 5 + 2      # profile up to 5-hops (plus self, plus unreachable)
        self.dim_1hot_ppr: int = 1
        self.dim_1hot_drnl: int = 25 + 1    # profile up to 25 = 5*5 different hop combinations (plus unreachable)
        self.profiler = SubgraphProfiler(self.num_ensemble, metrics=metrics_profile)
        

    def _get_cur_batch_size(self, mode):
        self.end_epoch[mode] = False
        return min(self.entity_epoch[mode].shape[0] - self.idx_entity_evaluated[mode], self.batch_size[mode])

    def _update_batch_stat(self, mode, batch_size):
        self.batch_num += 1
        self.idx_entity_evaluated[mode] += batch_size
        # TODO: may move this to epoch_start reset. 
        # --> Then you don't need end_epoch. Just check from idx_entity_evaluated
        if self.idx_entity_evaluated[mode] >= self.entity_epoch[mode].shape[0]:
            assert self.idx_entity_evaluated[mode] == self.entity_epoch[mode].shape[0]
            self.idx_entity_evaluated[mode] = 0
            self.end_epoch[mode] = True
            if self.graph_sampler[mode] is not None:
                assert self.pool_subg[mode].num_subg == 0
                assert all(len(d) == 0 for d in self.pool_subg[mode].data)

    def shuffle_entity(self, mode):
        """
        YOU MUST CALL THIS FUNCTION BEFORE STARTING ANY EPOCH. 
        """
        if self.prediction_task == 'node':
            if self.graph_sampler[mode] is not None:    # no need to shuffle for full batch mode
                perm = np.random.permutation(self.raw_entity_set[mode].size)
                if self.percent_per_epoch[mode] < 1.0:
                    perm = perm[:int(np.ceil(self.percent_per_epoch[mode] * perm.size))]
                self.entity_epoch[mode] = self.raw_entity_set[mode][perm]
                self.label_epoch[mode] = self.label_full[self.entity_epoch[mode]]   # entity = node
                self.graph_sampler[mode].shuffle_targets(self.entity_epoch[mode])
        else:   # link prediction
            es = self.raw_entity_set[mode]
            if 'pos' in es and 'neg' in es:
                pos_edge, neg_edge = es['pos'], es['neg']
            else:
                assert 'pos' in es
                # TODO use valedge as input, then concat undirected valedges to be excluded in neg sample
                pos_edge = self.raw_entity_set[mode]['pos']
                all_train_edges = add_self_loops(to_undirected(torch.from_numpy(pos_edge).t()))[0]
                neg_edge = negative_sampling(
                    all_train_edges, num_nodes=self.adj[mode].indptr.size-1, num_neg_samples=pos_edge.shape[0]
                )
                neg_edge = neg_edge.t().numpy()
            edge_set = np.concatenate([pos_edge, neg_edge], axis=0)
            label_epoch = np.repeat([1, 0], [pos_edge.shape[0], neg_edge.shape[0]])[:, np.newaxis]
            perm = np.random.permutation(edge_set.shape[0])
            if self.percent_per_epoch[mode] < 1.0:
                perm = perm[:int(np.ceil(self.percent_per_epoch[mode] * perm.size))]
            self.entity_epoch[mode] = edge_set[perm]
            self.label_epoch[mode] = torch.from_numpy(label_epoch[perm]).to(self.dev_torch)
            # now the sampler (c++ / py) will traverse all end-points of the pos and neg edges
            # NOTE: keep entity_epoch as unflattened. let sampler flatten internally
            # TODO: set label_epoch for node task as well. 
            self.graph_sampler[mode].shuffle_targets(self.entity_epoch[mode])

    def epoch_start_reset(self, epoch, mode):
        """
        Reset structs so that sampled subgraphs can be properly reused later on.  
        """
        self.batch_num = -1
        if self.graph_sampler[mode] is None and self.mode_sample == self.SUBG:
            self.instantiate_sampler(*self.args_sampler_init, modes=[mode])
            if mode not in self.nocache_modes:
                self.record_subgraphs[mode] = [
                    "record" if g.name in REUSABLE_SAMPLER else "none" 
                    for g in self.graph_sampler[mode].sampler_list
                ]
            else:
                self.record_subgraphs[mode] = ['noncache'] * self.num_ensemble
        elif self.mode_sample == self.FULL:
            return    
        self.graph_sampler[mode].set_return_target_only([rs == 'reuse' for rs in self.record_subgraphs[mode]])     

    def is_end_epoch(self, mode):
        return self.end_epoch[mode]

    def epoch_end_reset(self, mode):
        self.end_epoch[mode] = False
        if self.mode_sample == self.FULL:
            return
        for i in range(len(self.record_subgraphs[mode])):
            if self.record_subgraphs[mode][i] == "record" and not self.cache_subg[mode].is_empty(i):
                self.record_subgraphs[mode][i] = "reuse"
        if (
            self.optm_level == 'high' 
            and self.mode_sample != self.FULL 
            and all(r == 'reuse' for r in self.record_subgraphs[mode])
        ):
            self.drop_full_graph_info(mode)
    
    def drop_full_graph_info(self, mode):
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
        for cfg in config_ensemble:
            assert "method" in cfg
            assert "size_root" not in cfg or cfg["size_root"] == 1
        config_ensemble_mode = {}
        for m in modes:
            self.batch_size[m] = sampler_config_ensemble_["batch_size"]
            config_ensemble_mode[m] = deepcopy(config_ensemble)
        # TODO: support different sampler config in val and test
        for m, cfg_l in config_ensemble_mode.items():
            if m in [VALID, TEST]:
                for cfg in cfg_l:
                    if cfg['method'] == 'ppr_st':
                        cfg['method'] = 'ppr'
        for cfg_mode, cfg_ensemble in config_ensemble_mode.items():
            for cfg in cfg_ensemble:
                cfg["size_root"] = 1 + (self.prediction_task == 'link')     # we want each target to have its own subgraph
                cfg["fix_target"] = True             # i.e., we differentiate root node from the neighbor nodes (compare with GraphSAINT)
                cfg["sequential_traversal"] = True   # (mode != "train")
                if self.prediction_task == 'link':
                    cfg['include_target_conn'] = False
                if cfg["method"] in ["ppr", 'ppr_st']:
                    cfg["type_"] = cfg_mode
                    cfg['name_data'] = self.name_data
                    cfg["dir_data"] = self.dir_data
                    cfg['is_transductive'] = self.is_transductive
                    if self.prediction_task == 'node':
                        _prep_target = self.raw_entity_set[cfg_mode]
                        _dup_modes = None
                    else:
                        _prep_target = np.arange(self.adj[TEST].indptr.size - 1)
                        _dup_modes = [TRAIN, VALID, TEST]
                    cfg['args_preproc'] = {'preproc_targets': _prep_target, 'duplicate_modes': _dup_modes}
            aug_feat_ens = [self.aug_feats] * len(cfg_ensemble)
            self.graph_sampler[cfg_mode] = GraphSamplerEnsemble(
                self.adj[cfg_mode], 
                self.feat_full[:, :self.dim_feat_raw], 
                cfg_ensemble, 
                aug_feat_ens, 
                max_num_threads=parallelism, 
                num_subg_per_batch=500, 
                bin_adj_files=bin_adj_files[cfg_mode], 
                seed_cpp=self.seed_cpp
            )
            self.is_stochastic_sampler[cfg_mode] = self.graph_sampler[cfg_mode].is_stochastic

    def par_graph_sample(self, mode):
        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        """
        subg_ens_l_raw = self.graph_sampler[mode].par_sample_ensemble(self.prediction_task)
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
        To improve the computation efficiency, we concatenate the batch number of individual adj into a single big adj. 
        i.e., the resulting adj is of the block-diagnol form. 
        
        Such concatenation does not increase computation complexity (since adj is in CSR) but facilitates parallelism. 
        """
        if self.graph_sampler[mode] is None:    # [node pred only] no sampling, return the full graph. 
            self._update_batch_stat(mode, self.batch_size[mode])
            targets = self.raw_entity_set[mode]
            assert ret_raw_idx, "None subg mode should only be used in preproc!"
            return OneBatchSubgraph(
                [self.adj[mode]],
                [self.feat_full],
                self.label_full[targets],
                None,
                [targets],
                None,
                [np.arange(self.adj[mode].shape[0])]
            )
        batch_size_ = self._get_cur_batch_size(mode)
        while self.pool_subg[mode].num_subg < batch_size_:
            self.par_graph_sample(mode)
        if batch_size_ != self.batch_size[mode]:        # end of epoch
            assert batch_size_ < self.batch_size[mode] and self.pool_subg[mode].num_subg == batch_size_
            self.graph_sampler[mode].validate_epoch_end()
        adj_ens, feat_ens, target_ens, feat_aug_ens = [], [], [], []
        label_idx = None
        subgs_ens, size_subg_ens = self.pool_subg[mode].collate(batch_size_)
        size_subg_ens = torch.tensor(size_subg_ens).to(self.dev_torch)
        feat_aug_ens = []
        idx_b_start = self.idx_entity_evaluated[mode]
        idx_b_end = idx_b_start + batch_size_
        for subgs in subgs_ens:       
            assert subgs.target.size == batch_size_ * (1 + (self.prediction_task == 'link'))
            if label_idx is None:
                label_idx = subgs.node[subgs.target]
            else:
                assert np.all(label_idx == subgs.node[subgs.target])
            adj_ens.append(subgs.to_csr_sp())
            feat_ens.append(self.feat_full[subgs.node].to(self.dev_torch))
            target_ens.append(subgs.target)
            label_batch = self.label_epoch[mode][idx_b_start : idx_b_end].to(self.dev_torch)
            feat_aug_ens.append({})
            for candy_augs in {'hops', 'pprs', 'drnls'}.intersection(self.aug_feats):
                candy_aug = candy_augs[:-1] # remove 's'
                feat_aug_ens[-1][candy_augs] = getattr(subgs.entity_enc, f'{candy_aug}2onehot_vec')(
                    getattr(self, f'dim_1hot_{candy_aug}'), return_type='tensor'
                ).type(self.dtype).to(self.dev_torch)
        self._update_batch_stat(mode, batch_size_)
        ret = OneBatchSubgraph(
            adj_ens, feat_ens, label_batch, size_subg_ens, target_ens, feat_aug_ens
        )
        self.profiler.update_subgraph_batch(ret)
        self.profiler.profile()
        if ret_raw_idx:     # TODO: this should support ens as well. multiple subg should have the same raw target idx
            assert ret.num_ens == 1
            ret.idx_raw = [subgs.node for subgs in subgs_ens]
        return ret

    def disable_cache(self, mode):
        if self.mode_sample is not self.FULL:
            self.record_subgraphs[mode] = ['noncache'] * self.num_ensemble
            self.nocache_modes.add(mode)

    def get_aug_dim(self, aug_type):
        return getattr(self, f'dim_1hot_{aug_type[:-1]}')
