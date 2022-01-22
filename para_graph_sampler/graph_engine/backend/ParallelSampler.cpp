#include "ParallelSampler.h"
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


void ParallelSampler::drop_full_graph_info() {
  // TODO in ensemble, even if you cannot drop indices, you can still drop top_ppr_*
  if (graph_full.indices.size() > 0 || top_ppr_neighs.size() > 0 || top_ppr_scores.size() > 0) {
    std::vector<NodeType>().swap(graph_full.indices);
    std::vector<std::vector<NodeType>>().swap(top_ppr_neighs);
    std::vector<std::vector<PPRType>>().swap(top_ppr_scores);
    std::vector<float>().swap(graph_full.data);
    std::vector<float>().swap(edge_reweighted);
    // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
    // std::cout << "PERMANANTLY DELETING [GRAPH_FULL.INDICES, TOP_PPR_NEIGHS, TOP_PPR_SCORES] IN C++ SAMPLER" << std::endl;
    // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  }
}

void ParallelSampler::shuffle_targets(std::vector<NodeType> targets_pre_shuffled) {
  if (targets_pre_shuffled.size() == 0) {
    std::random_shuffle(nodes_target.begin(), nodes_target.end());
  } else {
    assert(nodes_target.size() == 0 || nodes_target.size() == targets_pre_shuffled.size());
    nodes_target.swap(targets_pre_shuffled);
  }
}

NodeType ParallelSampler::get_idx_root() {return idx_root;}

bool ParallelSampler::is_seq_root_traversal() {return sequential_traversal;}

NodeType ParallelSampler::num_nodes() {return graph_full.indptr.size() - 1;}

NodeType ParallelSampler::num_edges() {return graph_full.indices.size();}

NodeType ParallelSampler::num_nodes_target() {return nodes_target.size();}

void ParallelSampler::cleanup_history_subgraphs_ensemble() {
  for (auto& subgraphs : subgraphs_ensemble) {
    std::fill(subgraphs.indptr_vec.begin(), subgraphs.indptr_vec.end(), std::vector<NodeType>());
    std::fill(subgraphs.indices_vec.begin(), subgraphs.indices_vec.end(), std::vector<NodeType>());
    std::fill(subgraphs.data_vec.begin(), subgraphs.data_vec.end(), std::vector<float>());
    std::fill(subgraphs.origNodeID_vec.begin(), subgraphs.origNodeID_vec.end(), std::vector<NodeType>());
    std::fill(subgraphs.origEdgeID_vec.begin(), subgraphs.origEdgeID_vec.end(), std::vector<NodeType>());
    std::fill(subgraphs.target_vec.begin(), subgraphs.target_vec.end(), std::vector<NodeType>());
    std::fill(subgraphs.hop_vec.begin(), subgraphs.hop_vec.end(), std::vector<NodeType>());
    std::fill(subgraphs.ppr_vec.begin(), subgraphs.ppr_vec.end(), std::vector<PPRType>());
    std::fill(subgraphs.drnl_vec.begin(), subgraphs.drnl_vec.end(), std::vector<NodeType>());
    subgraphs.num_valid_subg_cur_batch = 0;
  }
}

void ParallelSampler::read_array_from_bin(std::string name_file, std::vector<NodeType> &ret) {
  if (name_file.length() == 0) {
    return;
  }
  double t11 = omp_get_wtime();
  std::ifstream file(name_file, std::ios::binary);
  file.unsetf(std::ios::skipws);
  std::streampos fileSize;
  file.seekg(0, std::ios::end);
  fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  auto num_elements = (NodeType)(fileSize / sizeof(NodeType));
  ret.resize(num_elements);
  file.read((char *)ret.data(), fileSize);
  file.close();
  double t22 = omp_get_wtime();
}

/*
 * Write the pre-computed PPR info into external binary file. 
 * Next time when running on the same graph with the same sampler
 * configuration, the sampler will read from the binary file rather
 * than computing PPR again. 
 */
void ParallelSampler::write_PPR_to_binary_file(
      std::string name_out_neighs, 
      std::string name_out_scores,
      int k, 
      float alpha, 
      float epsilon
) {
  if (name_out_neighs.length() == 0 || name_out_scores.length() == 0) {
    std::cout << "NOT writing PPR to output file! " << std::endl;
    return;
  }
  std::ofstream out_neighs(name_out_neighs, std::ios::out | std::ios::binary);
  if (out_neighs.is_open()) {
    out_neighs.write((char*)&alpha, sizeof(alpha));
    out_neighs.write((char*)&epsilon, sizeof(epsilon));
    out_neighs.write((char*)&k, sizeof(k));
    auto _size_full = static_cast<unsigned int>(top_ppr_neighs.size());
    out_neighs.write((char*)&_size_full, sizeof(_size_full));
    for (auto& nvec : top_ppr_neighs) {
      auto _size = static_cast<unsigned int>(nvec.size());
      // Consider new format (if memory becomes bottleneck): size, root, neigh1, neigh2, ...
      out_neighs.write((char*)&_size, sizeof(_size));
      for (auto nn : nvec) {
        out_neighs.write((char*)&nn, sizeof(nn));
      }
    }
    out_neighs.close();
  }
  std::ofstream out_scores(name_out_scores, std::ios::out | std::ios::binary);
  if (out_scores.is_open()) {
    out_scores.write((char*)&alpha, sizeof(alpha));
    out_scores.write((char*)&epsilon, sizeof(epsilon));
    out_scores.write((char*)&k, sizeof(k));
    auto _size_full = static_cast<unsigned int>(top_ppr_scores.size());
    out_scores.write((char*)&_size_full, sizeof(_size_full));
    for (auto& svec : top_ppr_scores) {
      auto _size = static_cast<unsigned int>(svec.size());
      out_scores.write((char*)&_size, sizeof(_size));
      for (auto ns : svec) {
        out_scores.write((char*)&ns, sizeof(ns));
      }
    }
    out_scores.close();
  }
  std::cout << "written ppr to files: " << name_out_neighs << " and " << name_out_scores << std::endl;
}

/*
 * Read PPR from the external binary file (written by the 
 * `write_PPR_to_binary_file` function. 
 */
bool ParallelSampler::read_PPR_from_binary_file(
      std::string name_in_neighs, 
      std::string name_in_scores,
      int k, 
      float alpha, 
      float epsilon
) {
  if (name_in_neighs.length() == 0 || name_in_scores.length() == 0) {
    return false;
  }
  std::ifstream fin_neighs(name_in_neighs, std::ios::in | std::ios::binary);
  std::ifstream fin_scores(name_in_scores, std::ios::in | std::ios::binary);
  if (fin_neighs.good() && fin_scores.good()) {
    if (fin_neighs.is_open()) {
      float alpha_ = -1.;
      float epsilon_ = -1.;
      int k_ = -1;
      unsigned int root_cnt = -1;
      fin_neighs.read((char*)&alpha_, sizeof(alpha_));
      fin_neighs.read((char*)&epsilon_, sizeof(epsilon_));
      fin_neighs.read((char*)&k_, sizeof(k_));
      if (alpha_ != alpha || epsilon_ > 1.1*epsilon || epsilon_ < 0.9*epsilon || k_ < k) {
        fin_neighs.close();
        return false;
      }
      std::cout << "meta data matches" << std::endl;
      fin_neighs.read((char*)&root_cnt, sizeof(root_cnt));
      for (unsigned int i = 0; i < root_cnt; i++) {
        assert(static_cast<NodeType>(root_cnt) == graph_full.num_nodes);
        unsigned int deg_;      // NOTE: if change dtype here, need to change the write function too. 
        fin_neighs.read((char*)&deg_, sizeof(deg_));
        auto deg_clip = std::min(static_cast<NodeType>(deg_), static_cast<NodeType>(k));
        for (NodeType j = 0; j < static_cast<NodeType>(deg_); j++) {
          NodeType nidx;
          fin_neighs.read((char*)&nidx, sizeof(nidx));
          if (j < deg_clip) {
            top_ppr_neighs[i].push_back(nidx);
          }
        }
      }
    } else {
      fin_neighs.close();
      return false;
    }
    fin_neighs.close();
    if (fin_scores.is_open()) {
      float alpha_ = -1.;
      float epsilon_ = -1.;
      int k_ = -1;
      unsigned int root_cnt = -1;
      fin_scores.read((char*)&alpha_, sizeof(alpha_));
      fin_scores.read((char*)&epsilon_, sizeof(epsilon_));
      fin_scores.read((char*)&k_, sizeof(k_));
      if (alpha_ != alpha || epsilon_ > 1.1*epsilon || epsilon_ < 0.9*epsilon || k_ < k) {
        fin_scores.close();
        std::fill(top_ppr_neighs.begin(), top_ppr_neighs.end(), std::vector<NodeType>());
        return false;
      }
      std::cout << "meta data matches" << std::endl;
      fin_scores.read((char*)&root_cnt, sizeof(root_cnt));
      for (unsigned int i = 0; i < root_cnt; i++) {
        assert(static_cast<NodeType>(root_cnt) == graph_full.num_nodes);
        unsigned int deg_;
        fin_scores.read((char*)&deg_, sizeof(deg_));
        auto deg_clip = std::min(deg_, static_cast<unsigned int>(k));
        for (unsigned int j = 0; j < deg_; j++) {
          PPRType nscore;
          fin_scores.read((char*)&nscore, sizeof(nscore));
          if (j < deg_clip) {
            top_ppr_scores[i].push_back(nscore);
            if (j > 0) {
              assert(top_ppr_scores[i][j-1] + 1e-25 >= top_ppr_scores[i][j]);
            }
          }
        }
      }
    } else {
      fin_scores.close();
      std::fill(top_ppr_scores.begin(), top_ppr_scores.end(), std::vector<PPRType>());
      return false;
    }
    fin_scores.close();
    return true;
  } else {
    return false;
  }
}


/*
 * approximate algorithm to compute PPR with given error budget
 */
void ParallelSampler::preproc_ppr_approximate(std::vector<NodeType>& preproc_target,
      int k, float alpha, float epsilon, std::string fname_neighs, std::string fname_scores) {
  auto n = graph_full.num_nodes;
  top_ppr_neighs.resize(n);
  top_ppr_scores.resize(n);
  alpha = 1 - alpha;
  // check if ppr vec / neighs have been computed and stored
  if (read_PPR_from_binary_file(fname_neighs, fname_scores, k, alpha, epsilon)) {
    std::cout << "LOADING PPR INFO FROM EXTERNAL FILE: " << fname_neighs << " and " << fname_scores << std::endl;
    return;
  }
  std::vector<NodeType> degree_vec = graph_full.get_degrees();
  // setup top_ppr_neighs
  std::cout << "START COMPUTING PPR SCORE FOR " << preproc_target.size() << " NODES" << std::endl << std::flush;
  double t1 = omp_get_wtime();
  bool use_map = n > 5000000 ? true : false;
  if (use_map) {std::cout << "-- USING MAP for PPR comp" << std::endl;} 
  else {std::cout << "== USING VEC for PPR comp" << std::endl;}
  #pragma omp parallel for
  for (int64_t i_para = 0; i_para < preproc_target.size(); i_para++) {    // some omp version will complain on unsigned counter
    auto target = preproc_target[i_para];
    std::unordered_map<NodeType, PPRType> touched_neigh_map;
    std::map<NodeType, PPRType> pi_eps_m;
    std::map<NodeType, PPRType> residue_m;
    std::vector<PPRType> pi_eps_v;
    std::vector<PPRType> residue_v;
    if (use_map) {
      pi_eps_m[target] = 0.;
      residue_m[target] = 1.;
    } else {
      pi_eps_v.resize(n, 0.);
      residue_v.resize(n, 0.);
      residue_v[target] = 1.;
    }
    std::set<NodeType> prop_set {target};
    while (prop_set.size() > 0) {
      auto v_prop = *(prop_set.begin());
      PPRType res_target_orig;
      if (use_map) {
        res_target_orig = residue_m[v_prop];
        if (pi_eps_m.find(v_prop) != pi_eps_m.end()) {
          pi_eps_m[v_prop] += alpha * res_target_orig;
        } else {
          pi_eps_m[v_prop] = alpha * res_target_orig;
        }
      } else {
        res_target_orig = residue_v[v_prop];
        pi_eps_v[v_prop] += alpha * res_target_orig;
      }
      auto m = (1 - alpha) * res_target_orig / (2 * degree_vec[v_prop]);
      for (NodeType i = graph_full.indptr[v_prop]; i < graph_full.indptr[v_prop + 1]; i ++) {
        auto u = graph_full.indices[i];
        if (use_map) {
          if (residue_m.find(u) != residue_m.end()) {
            residue_m[u] += m;
          } else {
            residue_m[u] = m;
          }
          if (residue_m[u] > epsilon * degree_vec[u]) {
            prop_set.insert(u);
          }
        } else {
          residue_v[u] += m;
          if (residue_v[u] > epsilon * degree_vec[u]) {
            prop_set.insert(u);
          }
        }
      }
      if (use_map) {
        residue_m[v_prop] = res_target_orig * (1 - alpha) / 2;
        if (residue_m[v_prop] <= epsilon * degree_vec[v_prop]) {
          prop_set.erase(v_prop);
          touched_neigh_map[v_prop] = pi_eps_m[v_prop];
        }
      } else {
        residue_v[v_prop] = res_target_orig * (1 - alpha) / 2;
        if (residue_v[v_prop] <= epsilon * degree_vec[v_prop]) {
          prop_set.erase(v_prop);
          touched_neigh_map[v_prop] = pi_eps_v[v_prop];
        }
      }
    }
    // co-sorting indices.
    NodeType _k = std::min(static_cast<NodeType>(k), static_cast<NodeType>(touched_neigh_map.size()));
    std::vector<std::pair<PPRType, NodeType>> pi_idx;
    pi_idx.reserve(touched_neigh_map.size());
    for (auto ni : touched_neigh_map) {
      pi_idx.push_back(std::make_pair(-ni.second, ni.first));
    }
    std::nth_element(pi_idx.begin(), pi_idx.begin() + _k, pi_idx.end());
    std::sort(pi_idx.begin(), pi_idx.begin() + _k);    // We need this to allow reuse of the vecs from other runs with smaller k
    // extract just the indices
    std::vector<NodeType> top_idx;
    std::vector<PPRType> top_score;
    for (NodeType i = 0; i < _k; i++) {
      top_idx.push_back(pi_idx[i].second);
      top_score.push_back(-pi_idx[i].first);
      if (i > 1 && -pi_idx[1].first == 0) {
        assert(-pi_idx[i].first == 0);
      }
    }
    top_ppr_neighs[target] = top_idx;
    top_ppr_scores[target] = top_score;
  }
  double t2 = omp_get_wtime();
  std::cout << "TIME FOR PPR: " << t2 - t1 << std::endl;
  write_PPR_to_binary_file(fname_neighs, fname_scores, k, alpha, epsilon);
}


/*
 * Return the induced subgraph from nodes selected by the specific sampler
 */
SubgraphStruct ParallelSampler::_node_induced_subgraph(std::unordered_map<NodeType, PPRType>& nodes_touched,
    std::vector<NodeType>& targets, bool include_self_conn, bool include_target_conn, std::set<std::string>& config_aug) {
  SubgraphStruct ret_subg_info;
  std::unordered_map<NodeType, NodeType> orig2subID;     // mapping from original graph node id to subgraph node id
  // first traversal to sort orig ID (potentially makes the python indexing faster)
  std::vector<std::pair<NodeType, PPRType>> temp_origNodeID;
  if (targets.size() == 1) {
    include_target_conn = true;
  }
  for (auto vp : nodes_touched) {
    temp_origNodeID.push_back(vp);      // NOTE: PPR val for the node will be useless if we have more than 1 target
  }
  std::sort(temp_origNodeID.begin(), temp_origNodeID.end());
  for (auto vp : temp_origNodeID) {
    ret_subg_info.origNodeID.push_back(vp.first);
    ret_subg_info.ppr.push_back(vp.second);
  }
  // second traversal to get the mapping
  NodeType cnt_subg_nodes = 0;
  for (auto v : ret_subg_info.origNodeID) {
    orig2subID[v] = cnt_subg_nodes;
    cnt_subg_nodes ++;
  }
  if (fix_target) {
    for (auto t : targets) {
      ret_subg_info.target.push_back(orig2subID[t]);
    }
  }
  // third traversal to build the neighbor list
  cnt_subg_nodes = 0;
  ret_subg_info.indptr.push_back(0);
  for (auto v : ret_subg_info.origNodeID) {
    auto idx_start = graph_full.indptr[v];
    auto idx_end = graph_full.indptr[v+1];
    ret_subg_info.indptr.push_back(0);
    NodeType idx_insert = -1;
    if (include_self_conn) {
      auto idx_self_upper = std::upper_bound(graph_full.indices.begin() + idx_start, 
                                            graph_full.indices.begin() + idx_end, v);
      auto idx_self_lower = std::lower_bound(graph_full.indices.begin() + idx_start, 
                                            graph_full.indices.begin() + idx_end, v);
      if (idx_self_upper == idx_self_lower) {
        idx_insert = idx_self_upper - graph_full.indices.begin();
      }
    }
    NodeType idx_end_adjusted = idx_insert >= 0 ? idx_end + 1 : idx_end;
    bool passed_self_e = false;
    for (NodeType e = idx_start; e < idx_end_adjusted; e++) {
      NodeType e_adjusted = passed_self_e ? e - 1 : e;
      auto neigh = graph_full.indices[e_adjusted];
      if (e == idx_insert) {
        passed_self_e = true;
        ret_subg_info.indices.push_back(orig2subID[v]);
        ret_subg_info.indptr[cnt_subg_nodes+1] ++;
        ret_subg_info.origEdgeID.push_back(-1);
        ret_subg_info.data.push_back(1.);
      } else if (nodes_touched.find(neigh) != nodes_touched.end() && 
          (include_target_conn || std::find(targets.begin(), targets.end(), v) == targets.end() 
                               || std::find(targets.begin(), targets.end(), neigh) == targets.end())) {
        ret_subg_info.indices.push_back(orig2subID[neigh]);
        ret_subg_info.indptr[cnt_subg_nodes+1] ++;
        ret_subg_info.origEdgeID.push_back(e_adjusted);
        ret_subg_info.data.push_back(1.);
      }
    }
    cnt_subg_nodes ++;
  }
  // fix indptr for a valid CSR
  for (auto i = 0; i < cnt_subg_nodes; i++) {
    ret_subg_info.indptr[i+1] += ret_subg_info.indptr[i];
  }
  // augmentation
  if (config_aug.find("hops") != config_aug.end()) {
    assert(config_aug.find("drnls") == config_aug.end());
    ret_subg_info.compute_hops(-1);
  }
  // compute drnl
  if (config_aug.find("drnls") != config_aug.end()) {
    assert(config_aug.find("hops") == config_aug.end());
    std::vector<NodeType> dx;
    std::vector<NodeType> dy;
    ret_subg_info.compute_hops(0);
    dx.swap(ret_subg_info.hop);
    ret_subg_info.compute_hops(1);
    dy.swap(ret_subg_info.hop);
    auto num_subg_node = ret_subg_info.indptr.size() - 1;
    ret_subg_info.drnl.resize(num_subg_node);
    for (auto i = 0; i < dx.size(); i++) {
      ret_subg_info.drnl[i] = ret_subg_info.compute_drnl_single(dx[i], dy[i]);
    }
  }
  return ret_subg_info;
}


std::vector<std::vector<NodeType>> ParallelSampler::_get_roots_p(int num_roots) {
  std::vector<std::vector<NodeType>> targets;
  if (sequential_traversal) {
    auto size_target = num_nodes_target();
    NodeType idx_start = idx_root;
    NodeType idx_end = std::min(size_target, static_cast<NodeType>(idx_start + num_roots*num_sampler_per_batch)); 
    idx_root = idx_end == size_target ? 0 : idx_end;
    for (NodeType i = idx_start; i < idx_end; i++) {
      if (targets.size() == 0 || (targets[targets.size()-1]).size() == static_cast<unsigned int>(num_roots)) {
        targets.push_back(std::vector<NodeType>());
      }
      targets[targets.size()-1].push_back(nodes_target[i]);
    }
  } else {
    targets.resize(num_sampler_per_batch);
    #pragma omp parallel for
    for (int p = 0; p < num_sampler_per_batch; p++) {
      for (int r = 0; r < num_roots; r++) {
        auto cur_node = nodes_target[rand()%nodes_target.size()];
        targets[p].push_back(cur_node);
      }
    }
  }
  return targets;
}


bool ParallelSampler::_extract_bool_config(
  std::unordered_map<std::string, std::string>& config, 
  std::string key, bool def_val
) {
  bool ret = def_val;
  if (config.find(key) != config.end()) {
    if (config.at(key).compare("true") == 0
        || config.at(key).compare("True") == 0
        || config.at(key).compare("1") == 0) {
      ret = true;
    }
  }
  return ret;
}

SubgraphStruct ParallelSampler::nodeIID(
  std::vector<NodeType>& targets, 
  std::set<std::string>& config_aug
) {
  std::unordered_map<NodeType, PPRType> nodes_touched;
  for (auto node : targets) {
    nodes_touched.insert(std::make_pair(node, -1));
  }
  auto ret = _node_induced_subgraph(nodes_touched, targets, false, false, config_aug);
  return ret;
}

SubgraphStruct ParallelSampler::khop(
  std::vector<NodeType>& targets, 
  std::unordered_map<std::string, std::string>& config,
  std::set<std::string>& config_aug
) {
  int depth = std::stoi(config.at("depth"));
  int budget = std::stoi(config.at("budget"));
  double t1 = omp_get_wtime();
  std::vector<std::set<NodeType>> nodes_per_level;
  nodes_per_level.push_back(std::set<NodeType>());
  for (auto t : targets) {
    nodes_per_level[0].insert(t);
  }
  // traverse from roots
  for (int lvl = 0; lvl < depth; lvl++) {
    std::set<NodeType> nodes_frontier;
    for (auto v : nodes_per_level[lvl]) {
      NodeType deg = graph_full.indptr[v+1] - graph_full.indptr[v];
      if (deg <= budget || budget < 0) {
        for (NodeType i = graph_full.indptr[v]; i < graph_full.indptr[v+1]; i++) {
          nodes_frontier.insert(graph_full.indices[i]);
        }
      } else {
        for (int i = 0; i < budget; i++) {
          auto offset = rand()%deg;
          nodes_frontier.insert(graph_full.indices[graph_full.indptr[v] + offset]);
        }
      }
    }
    nodes_per_level.push_back(nodes_frontier);
  }
  // prepare nodes_touched
  std::unordered_map<NodeType, PPRType> nodes_touched;
  for (auto& nodes_layer : nodes_per_level) {
    for (auto node : nodes_layer) {
      nodes_touched.insert(std::make_pair(node, -1));
    }
  }
  double t2 = omp_get_wtime();
  auto add_self_edge = _extract_bool_config(config, "add_self_edge", false);
  auto include_target_conn = _extract_bool_config(config, "include_target_conn", false);
  auto ret = _node_induced_subgraph(nodes_touched, targets, add_self_edge, include_target_conn, config_aug);
  double t3 = omp_get_wtime();
  time_sampler_total += t2 - t1;
  time_induction_total += t3 - t2;
  return ret;
}


/*
 * PPR sampler (not the function for calculating PPR scores):
 * Return neighbors in the scope by top PPR scores relative to the target node. 
 * 
 * This is a deterministic sampler. See `ppr_stochastic` for a stochastic variant. 
 */
SubgraphStruct ParallelSampler::ppr(
  std::vector<NodeType>& targets, 
  std::unordered_map<std::string, std::string>& config,
  std::set<std::string>& config_aug
) {
  int k = std::stoi(config.at("k"));
  float threshold = std::stod(config.at("threshold"));
  std::unordered_map<NodeType, PPRType> nodes_touched;
  for (auto t : targets) {
    nodes_touched[t] = -1;
    int64_t size_all_ppr = top_ppr_neighs[t].size();
    int64_t size_neigh = std::min(static_cast<int64_t>(k), size_all_ppr);
    PPRType max_ppr = 0;
    if (size_neigh > 1) {
      max_ppr = top_ppr_scores[t][1];
    } else {
      nodes_touched[t] = top_ppr_scores[t][0];
    }
    for (int64_t i = 0; i < size_neigh; i++) {
      if (max_ppr == 0 || top_ppr_scores[t][i] / max_ppr < threshold) {
        break;
      }
      nodes_touched[top_ppr_neighs[t][i]] = top_ppr_scores[t][i];
    }
    assert(nodes_touched[t] >= 0);
  }
  auto add_self_edge = _extract_bool_config(config, "add_self_edge", false);
  bool include_target_conn = _extract_bool_config(config, "include_target_conn", false);
  auto ret = _node_induced_subgraph(nodes_touched, targets, add_self_edge, include_target_conn, config_aug);
  return ret;
}


/*
 * A stochastic version of the PPR sampler
 * The difference between `ppr` is that here we select a neighbor node into the scope
 * with probability proportional to the PPR score (rather than always take the top-k)
 */
SubgraphStruct ParallelSampler::ppr_stochastic(
  std::vector<NodeType>& targets,
  std::unordered_map<std::string, std::string>& config,
  std::set<std::string>& config_aug
) {
  int k = std::stoi(config.at("k"));
  float threshold = std::stod(config.at("threshold"));
  // determine the sample size (based on the threshold value)
  std::vector<int> sample_size;
  for (auto t : targets) {
    auto size_neigh = std::min(static_cast<int64_t>(k), static_cast<int64_t>(top_ppr_neighs[t].size()));
    // option 1: keep the subg sizes consistent among TRAIN, VALID, TEST
    PPRType max_ppr = size_neigh <= 1 ? 0 : top_ppr_scores[t][1];
    int cnt_target = 0;
    for (auto i = 0; i < size_neigh; i++) {
      cnt_target ++;
      if (max_ppr == 0 || top_ppr_scores[t][i] / max_ppr < threshold) {
        break;
      }
    }
    sample_size.push_back(cnt_target);
    // option 2: this might lead to larger training graph sizes. 
    // sample_size.push_back(size_neigh);    // we sample by dist anyways, so probably don't need the cut-off on threshold
  }
  // sample by the distribution defined by top_ppr_scores
  std::unordered_map<NodeType, PPRType> nodes_touched;
  int cnt_t = 0;
  for (auto t : targets) {
    // generate sample_size number of nodes according to the weight defined by top_ppr_scores[t]
    std::vector<std::pair<double, int>> sample_w_idx;    // randomly generated weight, element index
    int idx_sample = 0;
    for (auto s : top_ppr_scores[t]) {
      auto u = static_cast<double>(rand() / (RAND_MAX));      // random number between 0, 1
      sample_w_idx.push_back(std::make_pair(-std::pow(u, 1 / s), idx_sample));
      idx_sample ++;
    }
    std::nth_element(sample_w_idx.begin(), sample_w_idx.begin() + sample_size[cnt_t], sample_w_idx.end());
    for (auto i_sel = 0; i_sel < sample_size[cnt_t]; i_sel ++) {
      auto sel = sample_w_idx[i_sel];
      nodes_touched[top_ppr_neighs[t][sel.second]] = top_ppr_scores[t][sel.second];
    }
    cnt_t ++;
  }
  auto add_self_edge = _extract_bool_config(config, "add_self_edge", false);
  auto include_target_conn = _extract_bool_config(config, "include_target_conn", false);
  auto ret = _node_induced_subgraph(nodes_touched, targets, add_self_edge, include_target_conn, config_aug);
  return ret;
}


SubgraphStruct ParallelSampler::dummy_sampler(std::vector<NodeType>& targets) {
  SubgraphStruct ret;
  for (auto t : targets) {
    ret.origNodeID.push_back(t);
  }
  return ret;  
}


std::vector<SubgraphStructVec> ParallelSampler::parallel_sampler_ensemble(
     std::vector<std::unordered_map<std::string, std::string>> configs_samplers,
     std::vector<std::set<std::string>> configs_aug) {
  cleanup_history_subgraphs_ensemble();
  assert(configs_samplers.size() == subgraphs_ensemble.size());
  int num_roots = 0;
  for (auto& cfg : configs_samplers) {
    if (!num_roots) {num_roots = std::stoi(cfg.at("num_roots"));}
    else {assert(num_roots == std::stoi(cfg.at("num_roots")));}
  }
  std::vector<std::vector<NodeType>> targets = _get_roots_p(num_roots);
  int num_sampler_cur_batch = targets.size();
  int cnt_ensemble = 0;
  for (auto& cfg : configs_samplers) {
    if (cfg.find("method") == cfg.end()) {
      std::cerr << "[C++ parallel sampler]: need to have the 'method' key in the config" << std::endl;
      exit(1);
    }
    bool return_target_only = _extract_bool_config(cfg, "return_target_only", false);
    #pragma omp parallel for
    for (int p = 0; p < num_sampler_cur_batch; p++) {
      SubgraphStruct subgraph_new;
      if (!return_target_only) {
        if (cfg.at("method").compare("khop") == 0) {
          subgraph_new = khop(targets[p], cfg, configs_aug[cnt_ensemble]);
        } else if (cfg.at("method").compare("ppr") == 0) {
          subgraph_new = ppr(targets[p], cfg, configs_aug[cnt_ensemble]);
        } else if (cfg.at("method").compare("ppr_st") == 0) {
          subgraph_new = ppr_stochastic(targets[p], cfg, configs_aug[cnt_ensemble]);
        } else if (cfg.at("method").compare("nodeIID") == 0) {
          subgraph_new = nodeIID(targets[p], configs_aug[cnt_ensemble]);
        }
      } else {
        subgraph_new = dummy_sampler(targets[p]);
      }
      subgraphs_ensemble[cnt_ensemble].add_one_subgraph_vec(subgraph_new, p);
      #pragma omp atomic
      subgraphs_ensemble[cnt_ensemble].num_valid_subg_cur_batch += 1;
    }
    cnt_ensemble ++;
  }
  return subgraphs_ensemble;
}


PYBIND11_MODULE(ParallelSampler, m) {
  py::class_<ParallelSampler>(m, "ParallelSampler")
      .def(
        py::init<
          std::vector<NodeType>, 
          std::vector<NodeType>, 
          std::vector<float>, 
          int, 
          int, 
          bool, 
          bool, 
          std::vector<float>&, 
          int, 
          std::string, 
          std::string, 
          std::string, 
          int
        >()
      )
      .def("num_nodes", &ParallelSampler::num_nodes)
      .def("num_edges", &ParallelSampler::num_edges)
      .def("num_nodes_target", &ParallelSampler::num_nodes_target)
      .def("shuffle_targets", &ParallelSampler::shuffle_targets)
      .def("get_idx_root", &ParallelSampler::get_idx_root)
      .def("is_seq_root_traversal", &ParallelSampler::is_seq_root_traversal)
      .def("preproc_ppr_approximate", &ParallelSampler::preproc_ppr_approximate)
      .def("parallel_sampler_ensemble", &ParallelSampler::parallel_sampler_ensemble)
      .def("drop_full_graph_info", &ParallelSampler::drop_full_graph_info);
  py::class_<SubgraphStructVec>(m, "SubgraphStructVec")
      .def("get_num_valid_subg", &SubgraphStructVec::get_num_valid_subg)
      .def("get_subgraph_indptr", &SubgraphStructVec::get_subgraph_indptr)
      .def("get_subgraph_indices", &SubgraphStructVec::get_subgraph_indices)
      .def("get_subgraph_data", &SubgraphStructVec::get_subgraph_data)
      .def("get_subgraph_node", &SubgraphStructVec::get_subgraph_node)
      .def("get_subgraph_edge_index", &SubgraphStructVec::get_subgraph_edge_index)
      .def("get_subgraph_target", &SubgraphStructVec::get_subgraph_target)
      .def("get_subgraph_hop", &SubgraphStructVec::get_subgraph_hop)
      .def("get_subgraph_ppr", &SubgraphStructVec::get_subgraph_ppr)
      .def("get_subgraph_drnl", &SubgraphStructVec::get_subgraph_drnl);
}
