## Assumption on graph data

* There can be many node types (e.g., total of 100 different types of nodes). 
* Number of neighbor types can be small (e.g., a node only has 2 types of neighbors). 
* Graphs are static. 

## Requirements

* Need to construct meta-path quickly (e.g., get all type-1 neighbors). 

## TODO
* heterogeneous adj list representation
    * hierarchical CSR with "map: node_type -> start_idx" as the indexing mechanism
* store node features as well and then share the data storage between frontend and backend