cmake_minimum_required(VERSION 3.9...3.18)
project(ParallelSampler)

find_package(OpenMP REQUIRED)
add_subdirectory(pybind11)
add_library(Graph STATIC Graph.cpp)
pybind11_add_module(ParallelSampler ParallelSampler.cpp)
target_link_libraries(ParallelSampler PRIVATE OpenMP::OpenMP_CXX PRIVATE Graph)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(ParallelSampler PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# ------------------------------------------------------
# the following flags may be necessary for some machines
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
