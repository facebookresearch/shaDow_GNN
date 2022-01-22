# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

setuptools.setup(
    name="para_graph_sampler",
    version="0.0.1",
    author="Hanqing Zeng",
    author_email="zhqhku@gmail.com",
    description="parallel graph engine for subgraph based GNN models",
    long_description="parallel graph engine for subgraph based GNN models",
    long_description_content_type="text/markdown",
    url="https://github.com/ZimpleX/para_graph_sampler",
    project_urls={
        "Bug Tracker": "https://github.com/ZimpleX/para_graph_sampler/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=["graph_engine", "graph_engine.frontend"],
    python_requires=">=3.8",
)