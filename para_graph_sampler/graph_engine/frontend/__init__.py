from graph_engine.frontend import graph
from graph_engine.frontend import (
    samplers_base,
    samplers_python,
    samplers_cpp,
    samplers_ensemble
)

# some constants
TRAIN = 0
VALID = 1
TEST  = 2

MODE2STR = {TRAIN: 'train', VALID: 'valid', TEST: 'test'}
STR2MODE = {'train': TRAIN, 'valid': VALID, 'test': TEST}