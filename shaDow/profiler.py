# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
from typing import Dict, Set

class SubgraphProfiler:
    MODES = {'running', 'global'}
    KNOWN_METRICS = {'hops'}
    QUEUE_SIZE = 5
    def __init__(self, num_ens, metrics: Dict[str, Set[str]]):
        self.num_ens = num_ens
        self.subgraph_batch = deque()
        assert set(metrics.keys()) == self.MODES
        assert len(metrics['running']) == 0 or set(metrics['running']).issubset(self.KNOWN_METRICS)
        assert len(metrics['global']) == 0 or set(metrics['global']).issubset(self.KNOWN_METRICS)
        self.metrics = metrics
        self.value_metrics = {
            md: [{m: [] for m in self.metrics[md]} for _ in range(num_ens)]
            for md in self.MODES
        }
        if len(metrics['running']) == 0:
            if len(metrics['global']) == 0:
                self.queue_size = 0
            else:
                self.queue_size = 1
        else:
            self.queue_size = self.QUEUE_SIZE
        
    def update_subgraph_batch(self, subgraph_batch) -> None:
        if self.queue_size == 0:
            return
        assert subgraph_batch.num_ens == self.num_ens
        if len(self.subgraph_batch) >= self.queue_size:
            self.subgraph_batch.popleft()
        self.subgraph_batch.append(subgraph_batch)

    def _profile_hops(
        self, hops: torch.Tensor, sizes_subg: torch.Tensor, mode: str
    ):
        if mode == 'global':
            offsets = torch.roll(torch.cumsum(sizes_subg, dim=0), 1)
            offsets[0] = 0
            idx = torch.arange(hops.shape[0], device=hops.device)
            return F.embedding_bag(idx, hops, offsets, mode='sum')
        elif mode == 'running':
            raise NotImplementedError
            # return hops.sum(axis=0) / sizes_subg.shape(0)
      
    def _summarize_hops(self, vals, mode: str):
        if mode == 'global':
            if len(vals) == 0:
                return []
            else:
                return torch.cat(vals).mean(axis=0)
        elif mode == 'running':
            # No need to profile the running value of subgraph hops (global is more accurate)
            return []

    def profile(self) -> None:
        sb = self.subgraph_batch[-1]
        for e in range(self.num_ens):
            if 'hops' in self.metrics['global']:
                if 'hops' not in sb.feat_aug_ens[e]:
                    break
                hops = sb.feat_aug_ens[e]['hops']
                self.value_metrics['global'][e]['hops'].append(
                    self._profile_hops(hops, sb.size_subg_ens[e], 'global')
                )
            
    def summarize(self):
        ret = {md: [{} for _ in range(self.num_ens)] for md in self.MODES}
        for md in self.MODES:
            for e in range(self.num_ens):
                for m in self.metrics[md]:
                    ret[md][e][m] = getattr(self, f'_summarize_{m}')(self.value_metrics[md][e][m], md)
                    self.value_metrics[md][e][m] = []      # clear after summarizing
        self.subgraph_batch = deque()
        return ret

    def _print_summary_hops(self, hop_stat: torch.Tensor, mode: str):
        if len(hop_stat) == 0:
            print("NO HOP STAT COLLECTED")
            return
        sep_str = "    "
        title = f"{'hops':>6s}{sep_str}"\
            + sep_str.join(f"{k:>6d}" for k in range(hop_stat.size(0) - 1))\
            + f"{sep_str}{'inf':>6s}"
        raw_vals = f"{'vals':>6s}{sep_str}"\
            + sep_str.join(f"{v:>6.2f}" for v in hop_stat[1:])\
            + f"{sep_str}{hop_stat[0].item():>6.2f}"
        denorm = (hop_stat[2:].sum() + hop_stat[0]).item()
        norm_vals = f"{'ratio':>6s}{sep_str}"\
            + f"{'--':>6s}{sep_str}"\
            + sep_str.join(f"{v/denorm*100:>6.2f}" for v in hop_stat[2:])\
            + f"{sep_str}{hop_stat[0].item()/denorm*100:>6.2f}"
        print("="*len(title))
        print(title)
        print("-"*len(title))
        print(raw_vals)
        print(norm_vals)
        print("="*len(title))
    
    def print_summary(self):
        if len(self.metrics['running']) > 0 or len(self.metrics['global']) > 0:
            str_title = "SUMMARY OF SUBG PROFILES"
            print("=" * len(str_title))
            print(str_title)
        ret = self.summarize()
        for md in self.MODES:
            for e in range(self.num_ens):
                for k, v in ret[md][e].items():
                    if k == 'hops':
                        self._print_summary_hops(v, md)
    
    def clear_metrics(self):
        self.metrics = {md: [] for md in self.MODES}