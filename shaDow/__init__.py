# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'            # for handling CTRL-C on Windows

# some constants
TRAIN = 0
VALID = 1
TEST  = 2

MODE2STR = {TRAIN: 'train', VALID: 'valid', TEST: 'test'}
STR2MODE = {'train': TRAIN, 'valid': VALID, 'test': TEST}
