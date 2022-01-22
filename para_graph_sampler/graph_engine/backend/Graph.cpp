// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <omp.h>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <utility>
#include <set>
#include <fstream>
#include <unordered_map>
#include <assert.h>
#include <stdlib.h>
#include <iterator>
#include <fstream>
#include <random>
#include "Graph.h"


std::vector<NodeType> GraphStruct::get_degrees() {
  std::vector<NodeType> deg(num_nodes, 0);
  #pragma omp parallel for
  for (int64_t i = 0; i < num_nodes; i ++) {  // some omp version will complain on unsigned counter
    deg[i] = indptr[i + 1] - indptr[i];
  }
  return deg;
}


void SubgraphStruct::compute_hops(int idx_target) {
  NodeType t;
  if (idx_target >= 0) {
    t = target[idx_target];
  } else {
    assert(target.size() == 1);
    t = target[0];
  }
  assert(indptr.size() > 0);    // only when subg is induced
  unsigned int N = indptr.size() - 1;
  hop.resize(N);
  std::fill(hop.begin(), hop.end(), -1);
  // BFS
  assert(static_cast<unsigned int>(t) < N);
  std::vector<bool> visited(N, false);
  std::queue<std::pair<NodeType, int>> q;
  visited[t] = true;
  q.push(std::make_pair(t, 0));
  while (!q.empty()) {
    auto frontier = q.front();
    NodeType cur_node = frontier.first;
    auto cur_hop = frontier.second;
    q.pop();
    hop[cur_node] = cur_hop;
    for (NodeType i = indptr[cur_node]; i < indptr[cur_node + 1]; i ++) {
      NodeType v_temp = indices[i];
      if (!visited[v_temp]) {
        q.push(std::make_pair(v_temp, cur_hop + 1));
        visited[v_temp] = true;
      }
    }
  }
}

NodeType SubgraphStruct::compute_drnl_single(NodeType dxi, NodeType dyi) {
  if (dxi >= 255 || dyi >= 255) {
    return 255;
  }
  NodeType di = dxi + dyi;
  auto ret = 1 + std::min(dxi, dyi) + (di / 2) * ((di / 2) + (di % 2) - 1);
  return ret;
}

/* NOTE
 * This function only adds the vector structure of the newly sampled subgraph. It does NOT
 * update the counter of num_valid_subg_cur_batch (as this counter needs to be updated atomicly)
 */
void SubgraphStructVec::add_one_subgraph_vec(SubgraphStruct& subgraph, int p) {
  indptr_vec[p] = subgraph.indptr;
  indices_vec[p] = subgraph.indices;
  data_vec[p] = subgraph.data;
  origNodeID_vec[p] = subgraph.origNodeID;
  origEdgeID_vec[p] = subgraph.origEdgeID;
  target_vec[p] = subgraph.target;
  hop_vec[p] = subgraph.hop;
  ppr_vec[p] = subgraph.ppr;
  drnl_vec[p] = subgraph.drnl;
}


int SubgraphStructVec::get_num_valid_subg() {
  return num_valid_subg_cur_batch;      // this is for the python interface to slice the *_vec structures
}
const std::vector<std::vector<NodeType>>& SubgraphStructVec::get_subgraph_indptr() {
  return indptr_vec;
}
const std::vector<std::vector<NodeType>>& SubgraphStructVec::get_subgraph_indices() {
  return indices_vec;
}
const std::vector<std::vector<float>>& SubgraphStructVec::get_subgraph_data() {
  return data_vec;
}
const std::vector<std::vector<NodeType>>& SubgraphStructVec::get_subgraph_node() {
  return origNodeID_vec;
}
const std::vector<std::vector<NodeType>>& SubgraphStructVec::get_subgraph_edge_index() {
  return origEdgeID_vec;
}
const std::vector<std::vector<NodeType>>& SubgraphStructVec::get_subgraph_target() {
  return target_vec;
}
const std::vector<std::vector<NodeType>>& SubgraphStructVec::get_subgraph_hop() {
  return hop_vec;
}
const std::vector<std::vector<PPRType>>& SubgraphStructVec::get_subgraph_ppr() {
  return ppr_vec;
}
const std::vector<std::vector<NodeType>>& SubgraphStructVec::get_subgraph_drnl() {
  return drnl_vec;
}
