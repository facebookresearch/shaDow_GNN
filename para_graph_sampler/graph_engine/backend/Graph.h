// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <stdint.h>

typedef uint32_t NodeType;    // work for as large as ogbn-papers100M
typedef float PPRType;

class GraphStruct{
 public:
  GraphStruct() {}
  GraphStruct(
    std::vector<NodeType> indptr_,
    std::vector<NodeType> indices_,
    std::vector<float> data_
  ) : indptr(indptr_), indices(indices_), data(data_) {
    num_nodes = indptr_.size() - 1;
    num_edges = indices_.size();
    assert(indptr_[num_nodes] == num_edges);
    assert(indptr_[0] == 0);
    std::cout << "NUM NODES: " << num_nodes << "\tNUM EDGES: " << num_edges << std::endl << std::flush;
  }
  std::vector<NodeType> get_degrees();
  std::vector<NodeType> indptr;
  std::vector<NodeType> indices;
  std::vector<float> data;
  NodeType num_nodes;
  NodeType num_edges;
};


class SubgraphStruct{
 public:
  SubgraphStruct() {}
  void compute_hops(int idx_target);    // compute hops to target, fill in hop vec
  NodeType compute_drnl_single(NodeType dxi, NodeType dyi);
  std::vector<NodeType> indptr;
  std::vector<NodeType> indices;
  std::vector<float> data;
  std::vector<NodeType> origNodeID;
  std::vector<NodeType> origEdgeID;
  std::vector<NodeType> target;
  // additional info to augment node feature
  std::vector<NodeType> hop;      // length = num subg nodes. 
  std::vector<PPRType> ppr;       // ppr for the single target
  std::vector<NodeType> drnl;
};

class SubgraphStructVec{
 public:
  SubgraphStructVec() {}
  SubgraphStructVec(int num_subgraphs) {
    indptr_vec.resize(num_subgraphs);
    indices_vec.resize(num_subgraphs);
    data_vec.resize(num_subgraphs);
    origNodeID_vec.resize(num_subgraphs);
    origEdgeID_vec.resize(num_subgraphs);
    target_vec.resize(num_subgraphs);
    hop_vec.resize(num_subgraphs);
    ppr_vec.resize(num_subgraphs);
    drnl_vec.resize(num_subgraphs);
  }
  std::vector<std::vector<NodeType>> indptr_vec;
  std::vector<std::vector<NodeType>> indices_vec;
  std::vector<std::vector<float>> data_vec;
  std::vector<std::vector<NodeType>> origNodeID_vec;
  std::vector<std::vector<NodeType>> origEdgeID_vec;
  std::vector<std::vector<NodeType>> target_vec;
  // additional info
  std::vector<std::vector<NodeType>> hop_vec;
  std::vector<std::vector<PPRType>> ppr_vec;
  std::vector<std::vector<NodeType>> drnl_vec;
  int num_valid_subg_cur_batch = 0;

  void add_one_subgraph_vec(SubgraphStruct& subgraph, int p);
  // getters
  int get_num_valid_subg();
  const std::vector<std::vector<NodeType>>& get_subgraph_indptr();
  const std::vector<std::vector<NodeType>>& get_subgraph_indices();
  const std::vector<std::vector<float>>& get_subgraph_data();
  const std::vector<std::vector<NodeType>>& get_subgraph_node();
  const std::vector<std::vector<NodeType>>& get_subgraph_edge_index();
  const std::vector<std::vector<NodeType>>& get_subgraph_target();
  const std::vector<std::vector<NodeType>>& get_subgraph_hop();
  const std::vector<std::vector<PPRType>>& get_subgraph_ppr();
  const std::vector<std::vector<NodeType>>& get_subgraph_drnl();
};
