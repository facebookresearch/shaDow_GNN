# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from sklearn import metrics
from ogb.nodeproppred import Evaluator

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

METRICS = { 'f1'            : ['f1mic', 'f1mac'],
            'accuracy'      : ['accuracy'],
            'accuracy_ogb'  : ['accuracy']}

class Metrics:
    full_graph_name = {'arxiv'      : 'ogbn-arxiv',
                       'products'   : 'ogbn-products',
                       'papers100M' : 'ogbn-papers100M'}
    def __init__(self, name_data, is_sigmoid : bool, metric : str):
        self.name_data = name_data
        self.is_sigmoid = is_sigmoid
        self.name = metric
        if metric == 'f1':
            self.calc = self._calc_f1
            self.is_better = self._is_better_f1
        elif metric == 'accuracy':
            self.calc = self._calc_accuracy
            self.is_better = self._is_better_accuracy
        elif metric == 'accuracy_ogb':
            self.evaluator = Evaluator(name=self.full_graph_name[name_data])
            self.calc = self._calc_accuracy_ogb
            self.is_better = self._is_better_accuracy
        else:
            raise NotImplementedError


    def _calc_f1(self, y_true, y_pred):
        """
        Compute F1-score (micro- and macro averaged for multiple classes).

        NOTE: for the case of each node having a single label (e.g., ogbn-arxiv),
            F1-micro score is equivalent to accuracy. 
        """
        if not self.is_sigmoid:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
        return {
            'f1mic' : metrics.f1_score(y_true, y_pred, average="micro"),
            'f1mac' : metrics.f1_score(y_true, y_pred, average="macro")
        }
 
    def _calc_accuracy(self, y_true, y_pred):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        # if each node has only 1 ground truth label, accuracy is equivalent to f1-micro
        return {
            'accuracy' : metrics.f1_score(y_true, y_pred, average="micro")
        }

    def _calc_accuracy_ogb(self, y_true, y_pred):
        """
        This function is equivalent to _calc_accuracy. We just do this to conform to the leaderboard requirement
        """
        y_true = np.argmax(y_true, axis=1)[:, np.newaxis]
        y_pred = np.argmax(y_pred, axis=1)[:, np.newaxis]
        acc = self.evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['acc']
        return {
            'accuracy' : acc
        }

    def _is_better_accuracy(self, loss_all, loss_min_hist, accuracy_all, accuracy_max_hist):
        acc_cur = accuracy_all[-1]
        return acc_cur > accuracy_max_hist

    def _is_better_f1(self, loss_all, loss_min_hist, f1mic_all, f1mic_max_hist, f1mac_all, f1mac_max_hist):
        f1mic_cur = f1mic_all[-1]
        return f1mic_cur > f1mic_max_hist
