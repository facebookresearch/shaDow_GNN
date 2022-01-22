# Graph Engine for Handling Large Scale Graph Analytics Tasks

This directory contains a stand-alone module for handling large scale graph analytics tasks. Subgraph sampling is one example of such tasks. Currently, shaDow-GNN is an example use-case of the graph engine. We will make this an individual repo in the near future. 

## Overview

`frontend/`: the `python` interface. Can be exposed to PyTorch trainer (e.g., to train shaDow-GNN). 
`backend/`: the parallel and optimized implementation of graph sampling algorithms. Also the internal data structure to store large scale graphs. 

## Install packages for graph engine sampler

```
bash install.sh
```
