# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import numpy as np
import torch
from typing import List, Dict
from graph_engine.frontend.samplers_cpp import (
    PPRSamplingCpp,
    PPRSTSamplingCpp,
    KHopSamplingCpp,
    NodeIIDCpp,
)
from graph_engine.frontend.samplers_python import KHopSamplingPy
from graph_engine.frontend.graph import EntityEncoding, Subgraph
import ParallelSampler as cpp_para_sampler


NAME2SAMPLER = {
    "ppr"       : ('cpp', PPRSamplingCpp),
    "ppr_st"    : ('cpp', PPRSTSamplingCpp),
    'ppr_feat_sim_inv': ('cpp', PPRSamplingCpp),
    "khop"      : ('cpp', KHopSamplingCpp), # ('python', KHopSamplingPy), 
    'nodeIID'   : ('cpp', NodeIIDCpp)
}

STOCHASTIC_SAMPLER = {'ppr_st', 'khop', 'feat_max_var'}

BACKEND_SUPPORTED_FEAT_AUG = {'hops', 'pprs', 'drnls'}

def find_all_backends(configs):
    return set([NAME2SAMPLER[c['method']][0] for c in configs])


@dataclass
class ConfigSampler:
    MODE2STR: Dict[int, str]        # {0: "train", 1: "valid", 2: "test"}
    fix_target: bool
    sequential_traversal: bool


class GraphSamplerEnsemble:
    def __init__(
        self,
        adj,
        feats,
        sampler_config_list : List[dict],
        aug_feat_list       : List[set],
        max_num_threads     : int,          # C++ / python MP
        num_subg_per_batch  : int=-1,
        bin_adj_files       : dict = None,  # C++
        seed_cpp            : int=-1,     # C++
        mode2str            : Dict[int, str] = {0: "train", 1: "valid", 2: "test"}
    ):
        """
        Supported backends: 
            C++, python
        We can have a mixure of backends. For example, if we are ensembling the
            PPR sampler with feature based sampler, then we can have PPR executed
            in C++ and feature based sampler executed in python. 
        """
        self.node_target = None
        # TODO: may not need the fix_target and sequential_traversal arguments
        _fix_target = [config.pop('fix_target') for config in sampler_config_list]
        _sequential_traversal = [config.pop('sequential_traversal') for config in sampler_config_list]
        assert len(set(_fix_target)) == 1 and len(set(_sequential_traversal)) == 1
        common_config = ConfigSampler(
            MODE2STR=mode2str, fix_target=_fix_target[0], sequential_traversal=_sequential_traversal[0]
        )
        for sc in sampler_config_list:
            sc["common_config"] = common_config
        assert num_subg_per_batch > 0 or max_num_threads > 0, \
            "You need to specify either sampler per batch OR num threads. "
        if num_subg_per_batch <= 0:
            num_subg_per_batch = int(max_num_threads * 10)
        edge_weights = self._reweight_graph_edges(sampler_config_list, adj, feats)
        all_backends = find_all_backends(sampler_config_list)
        cfg_backends = {b: [] for b in all_backends}
        aug_backends = {b: [] for b in all_backends}
        self.sampler_backends = {b: [] for b in all_backends}
        self.para_sampler = {b: None for b in all_backends}
        self.aug_feat = None
        for i_s, s in enumerate(sampler_config_list):       # split configs based on backend
            cfg_backends[NAME2SAMPLER[s['method']][0]].append(s)
            aug_backends[NAME2SAMPLER[s['method']][0]].append(
                aug_feat_list[i_s].intersection(BACKEND_SUPPORTED_FEAT_AUG)
            )
            if self.aug_feat is None:
                self.aug_feat = aug_feat_list[i_s]
            else:       # for now, assume all samplers perform the same set of feat aug
                assert self.aug_feat == aug_feat_list[i_s]
        assert common_config.sequential_traversal, \
            "non-sequential traversal does not support link prediction task and ensemble!"
        assert num_subg_per_batch % 2 == 0, "link task requires even number of subg per batch. "
        ###########################
        # instantiate C++ sampler #
        ###########################
        if 'cpp' in all_backends:
            args_Cpp_sampler = [
                num_subg_per_batch, 
                max_num_threads, 
                common_config.fix_target, 
                common_config.sequential_traversal, 
                edge_weights, 
                len(cfg_backends['cpp'])
            ]
            if bin_adj_files is None:
                args_Cpp_sampler = [adj.indptr, adj.indices, adj.data] + args_Cpp_sampler + ["", "", ""]
            else:
                args_Cpp_sampler = [[], [], []] + args_Cpp_sampler + [bin_adj_files[k] for k in ['indptr', 'indices', 'data']]
            args_Cpp_sampler += [seed_cpp]
            self.para_sampler['cpp'] = cpp_para_sampler.ParallelSampler(*args_Cpp_sampler)
            # adjust the order of creating the sampling instances. e.g., for PPR, 
            # since we are sharing the same C++ sampler, we need to load the pre-computed file with the largest k. 
            cfg_backends['cpp'], aug_backends['cpp'] = self._sort_sampler_order(cfg_backends['cpp'], aug_backends['cpp'])
            for ic, config in enumerate(cfg_backends['cpp']):
                _name = config.pop("method")
                config["para_sampler"] = self.para_sampler['cpp']       # share a common one
                self.sampler_backends['cpp'].append(NAME2SAMPLER[_name][1](adj, aug_backends['cpp'][ic], **config))
        ##############################
        # instantiate Python sampler #
        ##############################
        self.para_sampler['python'] = []
        if 'python' in all_backends:
            for ic, config in enumerate(cfg_backends['python']):
                _name = config.pop('method')
                if _name == 'feat_max_var':
                    config['feat_node'] = feats
                    config['num_proc'] = max_num_threads
                # TODO: unify config and common_config
                _sampler_inst = NAME2SAMPLER[_name][1](
                    adj, aug_backends['python'][ic], num_subg_per_batch=num_subg_per_batch, **config)
                self.sampler_backends['python'].append(_sampler_inst)
                self.para_sampler['python'].append(_sampler_inst)
        # check consistency among adj
        num_nodes_full, num_edges_full = [], []
        for b in all_backends:
            if type(self.para_sampler[b]) == list:
                num_nodes_full.extend(p.num_nodes() for p in self.para_sampler[b])
                num_edges_full.extend(p.num_edges() for p in self.para_sampler[b])
            else:
                num_nodes_full.append(self.para_sampler[b].num_nodes())
                num_edges_full.append(self.para_sampler[b].num_edges())
        assert len(set(num_nodes_full)) == 1 and len(set(num_edges_full)) == 1
        self.num_nodes_full = num_nodes_full.pop()
        self.num_edges_full = num_edges_full.pop()
        # It is up to the Minibatch class to determine if they only want targets
        self.return_target_only = {b: [None for _ in range(len(self.sampler_backends[b]))] for b in all_backends}
        # setup sampler list (frontend)
        self.sampler_list = self._merge_backend_lists(self.sampler_backends)
        assert set([s.name for s in self.sampler_list]).issubset(set(NAME2SAMPLER.keys()))
        self.is_stochastic = any(s.name in STOCHASTIC_SAMPLER for s in self.sampler_list)
    
    def _merge_backend_lists(self, dict_backends):
        return [vv for _k, v in dict_backends.items() for vv in v]

    def set_return_target_only(self, val):
        # TODO: handle python sampler
        cnt_backend = {b: 0 for b in set([s.backend for s in self.sampler_list])}
        for i, s in enumerate(self.sampler_list):
            b = s.backend
            self.return_target_only[b][cnt_backend[b]] = val[i]
            cnt_backend[b] += 1
        assert sum(cnt_backend.values()) == len(val) == len(self.sampler_list)

    def _reweight_graph_edges(self, config_list, adj, feats):
        """      configs for all backends
        Generate edge weights for sampling. 
        Assume we only generate one set of weights. 
        """
        return []

    def shuffle_targets(self, node_target_new):
        self.node_target = node_target_new.flatten()
        for _, p in self.para_sampler.items():
            if type(p) != list:
                # NOTE for C++ sampler, the py wrapper does not update the node_target here. 
                p.shuffle_targets(self.node_target)
            else:
                for pi in p:        # for python sampler
                    pi.shuffle_targets(self.node_target)

    def par_sample_ensemble(self, prediction_task):
        ret = {k: None for k in self.para_sampler.keys()}
        # handle C++ sampler first
        if 'cpp' in self.para_sampler:
            for i, cfg in enumerate(self.sampler_backends['cpp']):
                cfg.cpp_config['return_target_only'] = 'true' if self.return_target_only['cpp'][i] else 'false'
            _args = [cfg.cpp_config for cfg in self.sampler_backends['cpp']]
            _augs = [cfg.cpp_aug for cfg in self.sampler_backends['cpp']]
            ret['cpp'] = self.para_sampler['cpp'].parallel_sampler_ensemble(_args, _augs)
            for i, r in enumerate(ret['cpp']):
                ret['cpp'][i] = self._extract_subgraph_return(r, _args[i], _augs[i], not self.return_target_only['cpp'][i])
            assert len(ret['cpp']) == len(self.sampler_backends['cpp'])
            assert min([len(r) for r in ret['cpp']]) == max([len(r) for r in ret['cpp']])
        # handle python sampler
        if 'python' in self.para_sampler:
            ret['python'] = []
            for i, psam in enumerate(self.para_sampler['python']):
                ret['python'].append(psam.parallel_sample(return_target_only=self.return_target_only['python'][i]))
        ret_ens = self._merge_backend_lists(ret)
        ret_ens_merge = ret_ens
        # check consistency among samplers (regardless of backends)
        for s in zip(*ret_ens_merge):
            _root_idxs = [ss.node if ss.target.size == 0 else ss.node[ss.target] for ss in s]
            _num_roots = set([r.size for r in _root_idxs])
            assert len(_num_roots) == 1 and _num_roots.pop() == 1 + (prediction_task == 'link')
            assert len(set([r[0] for r in _root_idxs])) == 1
        return ret_ens_merge

    def _sort_sampler_order(self, sampler_config_list, aug_feat_list):
        """ Call this func for each backend. 
        While subgraph ensemble does not enforce an order of the samplers, 
            we "reorder" the samplers here so that C++ sampler knows which
            pre-computed PPR file to load from. 
        e.g., with a PPR of k=300 and a PPR of k=200, the C++ sampler would
            need to load the precomputed file of k=300. 
        """
        i_ppr_largest_k = None      # idx of the PPR sampler with largets k (idx in the original config_list)
        ppr_largest_k = None
        # assert ppr and ppr_stochastic do not coexist
        names_samplers = set(cfg['method'] for cfg in sampler_config_list)
        assert not ('ppr' in names_samplers and 'ppr_st' in names_samplers), \
            'pls check if you want both the determinstic and stochastic version of PPR?'
        f_ppr_k_factor = lambda is_ppr_st: 1 + is_ppr_st    # if stochastic PPR, then we sample k nodes from a pool of 2k candidates
        for i, cfg in enumerate(sampler_config_list):
            ppr_k_factor = f_ppr_k_factor(cfg['method'] == 'ppr_st')
            if cfg["method"] in ['ppr', 'ppr_st']:
                if i_ppr_largest_k is None or int(cfg["k"]) * ppr_k_factor > ppr_largest_k:
                    ppr_largest_k = int(cfg["k"]) * ppr_k_factor
                    i_ppr_largest_k = i
            cfg["is_preproc"] = False
        # re-order PPR
        if i_ppr_largest_k is not None:
            top1_ppr_sampler = sampler_config_list.pop(i_ppr_largest_k)
            if top1_ppr_sampler['method'] == 'ppr_st':
                top1_ppr_sampler['k_required'] = top1_ppr_sampler['k'] * f_ppr_k_factor(True)
            sampler_config_list = [top1_ppr_sampler] + sampler_config_list
            top1_aug_feat = aug_feat_list.pop(i_ppr_largest_k)
            aug_feat_list = [top1_aug_feat] + aug_feat_list
        # only preproc the first occurance of one type of sampler
        sampler_set = set()
        for cfg in sampler_config_list:
            if cfg["method"] not in sampler_set:
                sampler_set.add(cfg["method"])
                cfg["is_preproc"] = True
        return sampler_config_list, aug_feat_list

    def _extract_subgraph_return(self, ret_subg_struct, config_sampler, config_aug, validate_subg):
        """
        only for C++ backend
        """
        clip = ret_subg_struct.get_num_valid_subg()
        info = []
        for n in Subgraph.names_data_fields:
            r = getattr(ret_subg_struct, f'get_subgraph_{n}')()
            info.append([np.asarray(d) for d in r[:clip]])
        info_enc = []
        for n in EntityEncoding.names_data_fields:
            r = getattr(ret_subg_struct, f'get_subgraph_{n}')()
            if f"{n}s" not in config_aug:
                info_enc.append([np.array([]) for _ in range(clip)])
            else:
                info_enc.append([np.asarray(d) for d in r[:clip]])
        if config_sampler['method'] == 'ppr' and 'k' in config_sampler:
            cap_node_subg = int(config_sampler['k'])
            num_targets = set([tnp.size for tnp in info[Subgraph.names_data_fields.index('target')]])
            assert len(num_targets) == 1
            cap_node_subg *= num_targets.pop()
        else:
            cap_node_subg = self.num_nodes_full
        cap_edge_subg = min(self.num_edges_full, cap_node_subg**2)
        enc_batch = [
            EntityEncoding(
                cap_node_subg=cap_node_subg,
                cap_edge_subg=cap_edge_subg,
                validate=validate_subg,
                **dict(zip(EntityEncoding.names_data_fields, sie))
            ) for sie in zip(*info_enc)
        ]
        return [
            Subgraph(
                cap_node_full=self.num_nodes_full,
                cap_edge_full=self.num_edges_full,
                cap_node_subg=cap_node_subg,
                cap_edge_subg=cap_edge_subg,
                validate=validate_subg,
                entity_enc=enc_batch[i],
                **dict(zip(Subgraph.names_data_fields, si))
            ) for i, si in enumerate(zip(*info))
        ]

    def drop_full_graph_info(self):
        if 'cpp' in self.para_sampler:
            self.para_sampler['cpp'].drop_full_graph_info()

    def validate_epoch_end(self):
        if 'cpp' in self.para_sampler:
            assert self.para_sampler['cpp'].get_idx_root() == 0
            assert self.para_sampler['cpp'].is_seq_root_traversal()