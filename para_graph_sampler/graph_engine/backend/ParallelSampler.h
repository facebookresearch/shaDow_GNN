// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <stdint.h>
#include "Graph.h"

namespace py = pybind11;


class ParallelSampler{
 public:
  ParallelSampler(
    std::vector<NodeType> indptr_full_,
    std::vector<NodeType> indices_full_,
    std::vector<float> data_full_,
    int num_sampler_per_batch_,
    int max_num_threads_,
    bool fix_target_,
    bool sequential_traversal_,
    std::vector<float> edge_reweighted_=std::vector<float>(),
    int num_subgraphs_ensemble=1,
    std::string path_indptr="",
    std::string path_indices="",
    std::string path_data="",
    int seed=-1
  ) {
    if (indptr_full_.size() == 0) {
      read_array_from_bin(path_indptr, indptr_full_);
    }
    if (indices_full_.size() == 0) {
      read_array_from_bin(path_indices, indices_full_);
    }
    // NOTE TODO: right now we don't use any info from data array (we simply fill in 1. in subgraph)
    if (seed < 0) {
      std::srand(std::time(0));   // time(0) is equivalent to time(NULL)
    } else {
      std::srand(static_cast<unsigned int>(seed));
    }
    num_sampler_per_batch = num_sampler_per_batch_;   // this must be > 0
    if (max_num_threads_ <= 0) {                      // this can be <= 0, in which case we just let openmp decide 
      max_num_threads = omp_get_max_threads();
    } else {
      max_num_threads = max_num_threads_;
      omp_set_num_threads(max_num_threads);
      assert(max_num_threads == omp_get_max_threads());
    }
    fix_target = fix_target_;
    sequential_traversal = sequential_traversal_;
    graph_full = GraphStruct(indptr_full_, indices_full_, data_full_);
    edge_reweighted = edge_reweighted_;
    subgraphs_ensemble = std::vector<SubgraphStructVec>(
      num_subgraphs_ensemble, SubgraphStructVec(num_sampler_per_batch)
    );
  }
  // utilility functions
  NodeType num_nodes();
  NodeType num_edges();
  NodeType num_nodes_target();
  bool _extract_bool_config(std::unordered_map<std::string, std::string>& config, std::string key, bool def_val);
  // sampler: right now for all samplers, we return node-induced subgraph
  std::vector<SubgraphStructVec> parallel_sampler_ensemble(
    std::vector<std::unordered_map<std::string, std::string>> configs_samplers, 
    std::vector<std::set<std::string>> configs_aug
  );
  void preproc_ppr_approximate(
    std::vector<NodeType>& preproc_target, 
    int k, 
    float alpha, 
    float epsilon, 
    std::string fname_neighs, 
    std::string fname_scores
  );
  NodeType get_idx_root();             // for assertion of correctness
  bool is_seq_root_traversal();   // for assertion of correctness
  void shuffle_targets(std::vector<NodeType>);
  // ----------------
  // [DANGER] For aggressive optimization on memory consumption
  void drop_full_graph_info();

 private:
  void read_array_from_bin(std::string name_file, std::vector<NodeType> &ret);
  SubgraphStruct khop(
    std::vector<NodeType>& targets, 
    std::unordered_map<std::string, 
    std::string>& config, 
    std::set<std::string>& config_aug
  );   // config should specify k and budget
  SubgraphStruct ppr(
    std::vector<NodeType>& targets, 
    std::unordered_map<std::string, 
    std::string>& config, 
    std::set<std::string>& config_aug
  );    // config should specify k
  SubgraphStruct ppr_stochastic(
    std::vector<NodeType>& target, 
    std::unordered_map<std::string, std::string>& config, 
    std::set<std::string>& config_aug
  );
  SubgraphStruct nodeIID(
    std::vector<NodeType>& targets, 
    std::set<std::string>& config_aug
  );    // config doesn't need to specify anything
  SubgraphStruct dummy_sampler(std::vector<NodeType>& targets);
  void cleanup_history_subgraphs_ensemble();
  void write_PPR_to_binary_file(
    std::string name_out_neighs, 
    std::string name_out_scores, 
    int k, 
    float alpha, 
    float epsilon
  );
  bool read_PPR_from_binary_file(
    std::string name_in_neighs, 
    std::string name_in_scores, 
    int k, 
    float alpha, 
    float epsilon
  );

  std::vector<std::vector<NodeType>> _get_roots_p(int num_roots);
  SubgraphStruct _node_induced_subgraph(
    std::unordered_map<NodeType, PPRType>& nodes_touched, 
    std::vector<NodeType>& targets, 
    bool include_self_conn, 
    bool include_target_conn, 
    std::set<std::string>& config_aug
  );

  std::vector<NodeType> nodes_target;
  std::vector<std::vector<NodeType>> top_ppr_neighs;
  std::vector<std::vector<PPRType>> top_ppr_scores;      // may not essentially need it   // TODO may change it to be even more compact, since we only need to preserve relative values
  GraphStruct graph_full;
  std::vector<float> edge_reweighted;
  std::vector<SubgraphStructVec> subgraphs_ensemble;
  int num_sampler_per_batch;
  int max_num_threads;
  double time_sampler_total = 0;
  double time_induction_total = 0;
  bool fix_target;
  bool sequential_traversal;
  NodeType idx_root = 0;  // counter for sequential traversing the roots
                          // shared among all sampler ensembles
};
