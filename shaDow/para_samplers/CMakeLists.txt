cmake_minimum_required(VERSION 3.9...3.18)
project(ParallelSampler)

find_package(OpenMP REQUIRED)
add_subdirectory(pybind11)
pybind11_add_module(ParallelSampler ParallelSampler.cpp)
target_link_libraries(ParallelSampler PRIVATE OpenMP::OpenMP_CXX)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(ParallelSampler PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
