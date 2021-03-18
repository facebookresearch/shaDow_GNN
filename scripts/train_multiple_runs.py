# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import subprocess
import torch
from collections import defaultdict

"""
This is a small wrapper to repeat training multiple times (based on the OGB instructions)
"""


def parse_args():
    parser = argparse.ArgumentParser(description="repeat the same shaDow configuration multiple times")
    parser.add_argument("--dataset", required=True, type=str, help="name of data")
    parser.add_argument("--configs", required=True, type=str, default=None, help="path to the configuration of training (*.yml)")
    parser.add_argument("--log_test_convergence", type=int, default=1, help="how often to show the test accuracy during training")
    parser.add_argument("--gpu", default=None, type=int, help="which GPU to use")
    parser.add_argument("--nocache", type=str, default=None, choices=['train', 'valid', 'test', 'all'], help="don't caching the subgraph samples during for phase train / val / test")
    parser.add_argument("--full_tensor_on_gpu", action='store_true', required=False, help='if true, will leave the full graph node feat on gpu')
    # ===
    parser.add_argument("--repetition", required=True, type=int)
    return parser.parse_args()

args = parse_args()
cmd = f"python -m shaDow.main --configs {args.configs} --dataset {args.dataset} --log_test_convergence {args.log_test_convergence}"
if args.gpu is not None:
    cmd += f" --gpu {args.gpu}"
if args.nocache is not None:
    cmd += f" --nocache {args.nocache}"
if args.full_tensor_on_gpu:
    cmd += " --full_tensor_on_gpu"

cmd += " --no_pbar"


def run(args, cmd):
    for r in range(args.repetition):
        print(f"%%%%%%%%%%%%%%%%%%%%%")
        print(f"%  STARTING RUN {r:>2d}  %")
        print(f"%%%%%%%%%%%%%%%%%%%%%")
        print("Executing command: ")
        print(cmd)
        popen = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_lines in iter(popen.stdout.readline, ""):
            yield stdout_lines
        popen.stdout.close()

stats = {"train": defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}
for line in run(args, cmd):
    print(line, end="")
    if "FINAL SUMMARY: " in line:
        line = line.split("FINAL SUMMARY: ")[-1]
        line_fields = line.split()
        assert len(line_fields) % 3 == 0
        for i in range(0, len(line_fields), 3):
            stats[line_fields[i]][line_fields[i + 1]].append(float(line_fields[i + 2]))

print(f"FINAL STATUS OVER {args.repetition} RUNS:")
print(stats)
print("SUMMARY:")
for k, v in stats.items():
    for kk, vv in v.items():
        print(f"{k}, {kk} = ({torch.tensor(vv).float().mean().item()}, {torch.tensor(vv).float().std().item()})")
