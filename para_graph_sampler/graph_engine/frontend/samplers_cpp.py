# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
inheritance from base_graph_samplers: C++ version connecting to the `backend/`
"""

from __future__ import absolute_import, print_function

from typing import List
from graph_engine.frontend.samplers_base import (
    NodeIIDBase,
    KHopSamplingBase,
    PPRSamplingBase,
)
import ParallelSampler as cpp_para_sampler
import os
import glob


class NodeIIDCpp(NodeIIDBase):
    def __init__(
        self, 
        adj, 
        aug_feat,
        common_config,
        num_subg_per_batch=200, 
        para_sampler=None, 
        **kwargs
    ):
        super().__init__(adj, aug_feat, num_subg_per_batch=num_subg_per_batch)
        if not para_sampler:
            self.para_sampler = cpp_para_sampler.ParallelSampler(
                adj.indptr, adj.indices, adj.data, num_subg_per_batch, 
                common_config.fix_target, common_config.sequential_traversal
            )
        else:
            self.para_sampler = para_sampler
        self.cpp_config = {
            "method"        : "nodeIID",
            "num_roots"     : "1",
            "add_self_edge" : "false",
            "include_target_conn": "false"
        }
        self.cpp_aug = aug_feat
        self.backend = 'cpp'


class KHopSamplingCpp(KHopSamplingBase):
    def __init__(
        self, 
        adj, 
        aug_feat,
        size_root, 
        depth, 
        budget, 
        common_config,
        num_subg_per_batch=200, 
        para_sampler=None, 
        is_preproc=False, 
        add_self_edge=False,
        include_target_conn=False,
    ):
        super().__init__(adj, aug_feat, size_root, depth, budget, num_subg_per_batch=num_subg_per_batch)
        if not para_sampler:
            self.para_sampler = cpp_para_sampler.ParallelSampler(
                adj.indptr, adj.indices, adj.data, num_subg_per_batch, 
                common_config.fix_target, common_config.sequential_traversal
            )
        else:
            self.para_sampler = para_sampler
        self.cpp_config = {
            "method": "khop",
            "depth": str(depth), 
            "budget": str(budget), 
            "num_roots": str(size_root),
            "add_self_edge": "true" if add_self_edge else "false",
            "include_target_conn": "true" if include_target_conn else "false",
        }
        self.cpp_aug = aug_feat
        self.backend = 'cpp'


class PPRSamplingCpp(PPRSamplingBase):
    def __init__(
        self, 
        adj, 
        aug_feat,
        size_root, 
        k, 
        common_config,
        num_subg_per_batch: int=200, 
        alpha: float=0.85, 
        epsilon: float=1e-5, 
        threshold: float=0,
        type_: int=0, 
        name_data: str=None,        # used to identify stored preproc data
        dir_data: dict=None,        # 'local': xxx, 'remote': [yyy, zzz]
        is_transductive: bool=False,
        para_sampler=None,          # C++ backend
        is_preproc: bool=True, 
        add_self_edge: bool=False,
        include_target_conn: bool=False,
        args_preproc: dict={},
    ):
        if not para_sampler:
            self.para_sampler = cpp_para_sampler.ParallelSampler(
                adj.indptr, adj.indices, adj.data, num_subg_per_batch, 
                common_config.fix_target, common_config.sequential_traversal
            )
        else:
            self.para_sampler = para_sampler
        self.type = type_
        self.name_data = name_data
        self.dir_data = dir_data
        self.is_transductive = is_transductive
        self.is_preproc = is_preproc
        self.mode2str = common_config.MODE2STR
        epsilon = float(epsilon)
        alpha = float(alpha)
        super().__init__(adj, aug_feat, size_root, k, alpha=alpha, epsilon=epsilon, 
                threshold=threshold, args_preproc=args_preproc)
        self.cpp_config = {
            "method": "ppr",
            "k": str(k), 
            "num_roots": str(size_root),
            "threshold": str(threshold),
            "add_self_edge": "true" if add_self_edge else "false",
            "include_target_conn": "true" if include_target_conn else "false",
        }
        self.cpp_aug = aug_feat
        self.backend = 'cpp'

    def is_PPR_file_exists(self, file_mode):
        """
        Check if there exist files for previously computed PPR values. 
        """
        assert file_mode in self.mode2str.keys()
        if self.dir_data is None or self.dir_data['is_adj_changed']:
            return False, "", ""        # This will make C++ to compute PPR without storing it in file
        str_trans = "transductive" if self.is_transductive else "inductive"
        dir_data_local = self.dir_data['local']
        folder = dir_data_local
        dir_prefix = f"{self.name_data}/ppr_float"
        dir_suffix = f"{str_trans}_{self.mode2str[file_mode]}_{self.alpha}_{self.epsilon}"
        # ==
        fname_neighs_all = f"{folder}/{dir_prefix}/neighs_{dir_suffix}_*"
        fname_scores_all = f"{folder}/{dir_prefix}/scores_{dir_suffix}_*"
        fname_neighs_pattern = f"{folder}/{dir_prefix}/neighs_{dir_suffix}" + "_{}.bin"
        fname_scores_pattern = f"{folder}/{dir_prefix}/scores_{dir_suffix}" + "_{}.bin"
        k_required = self.k if not hasattr(self, 'k_required') else self.k_required
        fname_neighs = f"{folder}/{dir_prefix}/neighs_{dir_suffix}_{k_required}.bin"
        fname_scores = f"{folder}/{dir_prefix}/scores_{dir_suffix}_{k_required}.bin"
        # check if meet the condition
        candy_neighs = sorted(glob.glob(fname_neighs_all))      # will return [] if dir `ppr` doesn't exist
        candy_scores = sorted(glob.glob(fname_scores_all))
        is_found = False
        for cn in candy_neighs:
            _, __, k_meta = cn.split("/")[-1].split(".bin")[0].split("_")[-3:]
            if k_required <= int(k_meta):
                fname_neighs = fname_neighs_pattern.format(k_meta)
                fname_scores = fname_scores_pattern.format(k_meta)
                if fname_scores in candy_scores:
                    is_found = True
                    break
        dir_target = f"{dir_data_local}/{self.name_data}/ppr_float"
        if not os.path.exists(dir_target):
            os.makedirs(dir_target)
        return is_found, fname_neighs, fname_scores

    def preproc(self, preproc_targets, duplicate_modes=None):
        if not self.is_preproc:
            return
        is_found_self_type, fname_neighs, fname_scores = self.is_PPR_file_exists(self.type)
        if not is_found_self_type and duplicate_modes is not None and len(duplicate_modes) > 0:
            for m in duplicate_modes:
                if m == self.type:
                    continue
                is_found_m, fname_neighs, fname_scores = self.is_PPR_file_exists(m)
                if is_found_m:      # TODO: symb link to calculated ppr files, check fname_*
                    # breakpoint()
                    break
        self.para_sampler.preproc_ppr_approximate(
            preproc_targets, self.k, self.alpha, self.epsilon, fname_neighs, fname_scores
        )


class PPRSTSamplingCpp(PPRSamplingCpp):
    def __init__(
        self, 
        adj, 
        aug_feat,
        size_root, 
        k, 
        k_required,
        common_config,
        num_subg_per_batch: int=200, 
        alpha: float=0.85, 
        epsilon: float=1e-5, 
        threshold: float=0,
        type_: int=0, 
        name_data: str=None,        # used to identify stored preproc data
        dir_data: dict=None,        # 'local': xxx, 'remote': [yyy, zzz]
        is_transductive: bool=False,
        para_sampler=None,          # C++ backend
        is_preproc: bool=True, 
        add_self_edge: bool=False,
        include_target_conn: bool=False,
        args_preproc: dict={}
    ):
        self.k_required = k_required
        super().__init__(
            adj, 
            aug_feat, 
            size_root, 
            k, 
            common_config, 
            num_subg_per_batch=num_subg_per_batch, 
            alpha=alpha, 
            epsilon=epsilon, 
            threshold=threshold, 
            type_=type_, 
            name_data=name_data, 
            dir_data=dir_data, 
            is_transductive=is_transductive, 
            para_sampler=para_sampler, 
            is_preproc=is_preproc, 
            add_self_edge=add_self_edge,
            args_preproc=args_preproc
        )
        self.cpp_config = {
            "method": "ppr_st",
            "k": str(k), 
            "num_roots": str(size_root),
            "threshold": str(threshold),
            "add_self_edge": "true" if add_self_edge else "false",
            "include_target_conn": "true" if include_target_conn else "false",
        }
        self.name = 'ppr_st'
        self.backend = 'cpp'
