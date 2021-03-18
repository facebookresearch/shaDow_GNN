# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, print_function

from typing import List
from shaDow import TRAIN, VALID, TEST, MODE2STR
from shaDow.para_samplers.base_graph_samplers import (
    NodeIIDBase,
    KHopSamplingBase,
    PPRSamplingBase,
    Subgraph
)
try:
    import ParallelSampler as cpp_para_sampler
except Exception:       # two ways of building parallel sampler. The first way is recommended
    import shaDow.para_samplers.ParallelSampler as cpp_para_sampler
import numpy as np
import os
import glob


class NodeIID(NodeIIDBase):
    def __init__(
            self, 
            adj, 
            node_target,
            aug_feat,
            fix_target=True, 
            sequential_traversal=True, 
            num_subg_per_batch=200, 
            para_sampler=None, 
            **kwargs
        ):
        super().__init__(adj, node_target, aug_feat)
        if not para_sampler:
            self.para_sampler = cpp_para_sampler.ParallelSampler(
                adj.indptr, adj.indices, adj.data, node_target, 
                num_subg_per_batch, fix_target, sequential_traversal
            )
        else:
            self.para_sampler = para_sampler
        self.cpp_config = {
            "method"        : "nodeIID",
            "num_roots"     : "1",
            "add_self_edge" : "false"
        }
        self.cpp_aug = aug_feat

class KHopSampling(KHopSamplingBase):
    def __init__(
            self, 
            adj, 
            node_target, 
            aug_feat,
            size_root, 
            depth, 
            budget, 
            fix_target=True, 
            sequential_traversal=True, 
            num_subg_per_batch=200, 
            para_sampler=None, 
            is_preproc=False, 
            add_self_edge=False
        ):
        super().__init__(adj, node_target, aug_feat, size_root, depth, budget)
        if not para_sampler:
            self.para_sampler = cpp_para_sampler.ParallelSampler(
                adj.indptr, adj.indices, adj.data, node_target, 
                num_subg_per_batch, fix_target, sequential_traversal
            )
        else:
            self.para_sampler = para_sampler
        self.cpp_config = {
            "method": "khop",
            "depth": str(depth), 
            "budget": str(budget), 
            "num_roots": str(size_root),
            "add_self_edge": "true" if add_self_edge else "false",
        }
        self.cpp_aug = aug_feat


class PPRSampling(PPRSamplingBase):
    def __init__(
            self, 
            adj, 
            node_target, 
            aug_feat,
            size_root, 
            k, 
            fix_target : bool=True, 
            sequential_traversal : bool=True, 
            num_subg_per_batch : int=200, 
            alpha : float=0.85, 
            epsilon : float=1e-5, 
            threshold : float=0,
            type_ : int=TRAIN, 
            name_data : str=None,       # used to identify stored preproc data
            dir_data : dict=None,       # 'local': xxx, 'remote': [yyy, zzz]
            is_transductive : bool=False,
            para_sampler=None, 
            is_preproc : bool=True, 
            add_self_edge : bool=False
        ):
        if not para_sampler:
            self.para_sampler = cpp_para_sampler.ParallelSampler(
                adj.indptr, adj.indices, adj.data, node_target,
                num_subg_per_batch, fix_target, sequential_traversal
            )
        else:
            self.para_sampler = para_sampler
        self.type = type_
        self.name_data = name_data
        self.dir_data = dir_data
        self.is_transductive = is_transductive
        self.is_preproc = is_preproc
        epsilon = float(epsilon)
        alpha = float(alpha)
        super().__init__(adj, node_target, aug_feat, size_root, k, alpha=alpha, epsilon=epsilon, threshold=threshold)
        self.cpp_config = {
            "method": "ppr",
            "k": str(k), 
            "num_roots": str(size_root),
            "threshold": str(threshold),
            "add_self_edge": "true" if add_self_edge else "false",
        }
        self.cpp_aug = aug_feat

    def is_PPR_file_exists(self, file_mode):
        """
        Check if there exist files for previously computed PPR values. 
        """
        assert file_mode in [TRAIN, VALID, TEST]
        if self.dir_data is None or self.dir_data['is_adj_changed']:
            return False, "", ""        # This will make C++ to compute PPR without storing it in file
        str_trans = "transductive" if self.is_transductive else "inductive"
        dir_data_local = self.dir_data['local']
        folder = dir_data_local
        dir_prefix = f"{self.name_data}/ppr_float"
        dir_suffix = f"{str_trans}_{MODE2STR[file_mode]}_{self.alpha}_{self.epsilon}"
        # ==
        fname_neighs_all = f"{folder}/{dir_prefix}/neighs_{dir_suffix}_*"
        fname_scores_all = f"{folder}/{dir_prefix}/scores_{dir_suffix}_*"
        fname_neighs_pattern = f"{folder}/{dir_prefix}/neighs_{dir_suffix}" + "_{}.bin"
        fname_scores_pattern = f"{folder}/{dir_prefix}/scores_{dir_suffix}" + "_{}.bin"
        fname_neighs = f"{folder}/{dir_prefix}/neighs_{dir_suffix}_{self.k}.bin"
        fname_scores = f"{folder}/{dir_prefix}/scores_{dir_suffix}_{self.k}.bin"
        # check if meet the condition
        candy_neighs = sorted(glob.glob(fname_neighs_all))      # will return [] if dir `ppr` doesn't exist
        candy_scores = sorted(glob.glob(fname_scores_all))
        is_found = False
        for cn in candy_neighs:
            _, __, k_meta = cn.split("/")[-1].split(".bin")[0].split("_")[-3:]
            if self.k <= int(k_meta):
                fname_neighs = fname_neighs_pattern.format(k_meta)
                fname_scores = fname_scores_pattern.format(k_meta)
                if fname_scores in candy_scores:
                    is_found = True
                    break
        dir_target = f"{dir_data_local}/{self.name_data}/ppr_float"
        if not os.path.exists(dir_target):
            os.makedirs(dir_target)
        return is_found, fname_neighs, fname_scores

    def preproc(self):
        if not self.is_preproc:
            return
        _, fname_neighs, fname_scores = self.is_PPR_file_exists(self.type)
        self.para_sampler.preproc_ppr_approximate(self.k, self.alpha, self.epsilon, fname_neighs, fname_scores)

    



class GraphSamplerEnsemble:
    def __init__(
            self, 
            adj, 
            node_target, 
            sampler_config_list     : List[dict],
            aug_feat_list           : List[set],
            max_num_threads         : int,
            num_subg_per_batch      : int=-1,      # TODO: add num_threads
            bin_adj_files           : dict=None
    ):
        NAME_SAMPLER_MAPPING = {
            "ppr"   : PPRSampling,
            "khop"  : KHopSampling,
            'nodeIID': NodeIID,
        }
        self.node_target = node_target
        self.sampler_list = []
        fix_target = None
        sequential_traversal = None
        for config in sampler_config_list:
            if fix_target is None:
                fix_target = config["fix_target"]
            else:
                assert fix_target == config["fix_target"]
            if sequential_traversal is None:
                sequential_traversal = config["sequential_traversal"]
            else:
                assert sequential_traversal == config["sequential_traversal"]
        assert num_subg_per_batch > 0 or max_num_threads > 0, "You need to specify either sampler per batch OR num threads. "
        if num_subg_per_batch <= 0:
            num_subg_per_batch = int(max_num_threads * 10)
        args_Cpp_sampler = [node_target, num_subg_per_batch, max_num_threads, fix_target, sequential_traversal, len(sampler_config_list)]
        if bin_adj_files is None:
            args_Cpp_sampler = [adj.indptr, adj.indices, adj.data] + args_Cpp_sampler + ["", "", ""]
        else:
            args_Cpp_sampler = [[], [], []] + args_Cpp_sampler + [bin_adj_files['indptr'], bin_adj_files['indices'], bin_adj_files['data']]
        self.para_sampler = cpp_para_sampler.ParallelSampler(*args_Cpp_sampler)
        # adjust the order of creating the sampling instances. e.g., 
        # for PPR, since we are sharing the same C++ sampler, we need to load the pre-computed
        # file with the largest k. 
        sampler_config_list, aug_feat_list = self._sort_sampler_order(sampler_config_list, aug_feat_list)
        for ic, config in enumerate(sampler_config_list):
            _name = config.pop("method")
            config["para_sampler"] = self.para_sampler      # share a common one
            self.sampler_list.append(NAME_SAMPLER_MAPPING[_name](adj, node_target, aug_feat_list[ic], **config))
        self.cpp_config_list = [sm.cpp_config for sm in self.sampler_list]
        self.cpp_aug_list = [sm.cpp_aug for sm in self.sampler_list]
        self.num_nodes_full = self.para_sampler.num_nodes()
        self.num_edges_full = self.para_sampler.num_edges()
        
        self.return_target_only = None

    def shuffle_targets(self):
        self.para_sampler.shuffle_targets()

    def par_sample_ensemble(self):
        for i, sc in enumerate(self.cpp_config_list):
            sc["return_target_only"] = "true" if self.return_target_only[i] else "false"
        ret = self.para_sampler.parallel_sampler_ensemble(self.cpp_config_list, self.cpp_aug_list)
        for i, r in enumerate(ret):
            _cfg = self.cpp_config_list[i]      # to determine subg dtype
            _cfg_aug = self.cpp_aug_list[i]
            ret[i] = self._extract_subgraph_return(r, _cfg, _cfg_aug, not self.return_target_only[i])
        assert len(ret) == len(self.sampler_list)
        assert min([len(r) for r in ret]) == max([len(r) for r in ret])
        return ret

    def _sort_sampler_order(self, sampler_config_list, aug_feat_list):
        i_ppr_largest_k = None
        ppr_largest_k = None
        for i, cfg in enumerate(sampler_config_list):
            if cfg["method"] == "ppr":
                if i_ppr_largest_k is None or int(cfg["k"]) > ppr_largest_k:
                    ppr_largest_k = int(cfg["k"])
                    i_ppr_largest_k = i
            cfg["is_preproc"] = False
        # re-order PPR
        if i_ppr_largest_k is not None:
            top1_ppr_sampler = sampler_config_list.pop(i_ppr_largest_k)
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
        clip = ret_subg_struct.get_num_valid_subg()
        info = []
        for n in Subgraph.names_data_fields:
            r = getattr(ret_subg_struct, f'get_subgraph_{n}')()
            if (n == 'hop' or n == 'ppr') and f"{n}s" not in config_aug:
                info.append([np.array([]) for _ in range(clip)])
            else:
                info.append([np.asarray(d) for ir, d in enumerate(r) if ir < clip])
        if config_sampler['method'] == 'ppr' and 'k' in config_sampler:
            cap_node_subg = int(config_sampler['k'])
        else:
            cap_node_subg = self.num_nodes_full
        return [Subgraph(
                        cap_node_full=self.num_nodes_full,
                        cap_edge_full=self.num_edges_full,
                        cap_node_subg=cap_node_subg,
                        cap_edge_subg=min(self.num_edges_full, cap_node_subg**2),
                        validate=validate_subg, 
                        **dict(zip(Subgraph.names_data_fields, si))
                    ) 
                    for si in zip(*info)]

    def drop_full_graph_info(self):
        self.para_sampler.drop_full_graph_info()
