cd shaDow/para_samplers/
c++ -fopenmp -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` ParallelSampler.cpp -o ParallelSampler`python3-config --extension-suffix`
cd ../..
