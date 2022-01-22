import numpy as np
from sklearn import metrics
from ogb.nodeproppred import Evaluator as Evaluator_n
from ogb.linkproppred import Evaluator as Evaluator_l

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

METRICS = {
    'f1'            : ['f1mic', 'f1mac'],
    'accuracy'      : ['accuracy'],
    'accuracy_ogb'  : ['accuracy'],
    'hits20'        : ['hits20'], # 'hits50', 'hits100'],
    'hits50'        : ['hits50'], # 'hits20', 'hits100'],
    'hits100'       : ['hits100'] # 'hits50', 'hits20']}
}

class Metrics:
    full_graph_name = {
        'arxiv'      : 'ogbn-arxiv',
        'products'   : 'ogbn-products',
        'papers100M' : 'ogbn-papers100M',
        'collab'     : 'ogbl-collab',
        'ppa'        : 'ogbl-ppa'
    }
    def __init__(self, name_data: str, is_sigmoid: bool, metric: str, metric_win_size: int):
        self.window_size = metric_win_size
        self.name_data = name_data
        self.is_sigmoid = is_sigmoid
        self.name = metric
        if metric == 'f1':
            self.calc = self._calc_f1
            self.is_better = self._is_better_f1
            self.metric_term = ('f1mic', 'max')      # if terminate by multiple metrics, then [('f1mic', 'max'), ('f1mac', 'max')]
        elif metric == 'accuracy':
            self.calc = self._calc_accuracy
            self.is_better = self._is_better_accuracy
            self.metric_term = ('accuracy', 'max')
        elif metric == 'accuracy_ogb':
            self.evaluator = Evaluator_n(name=self.full_graph_name[name_data])
            self.calc = self._calc_accuracy_ogb
            self.is_better = self._is_better_accuracy
            self.metric_term = ('accuracy', 'max')
        elif metric.startswith('hits'):
            self.evaluator = Evaluator_l(name=self.full_graph_name[name_data])
            self.calc = self._calc_hits
            self.is_better = eval(f"self._is_better_{metric}")
            self.metric_term = (metric, 'max')
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
    
    def _calc_hits(self, y_true, y_pred):
        pos_pred = y_pred[y_true==1]
        neg_pred = y_pred[y_true==0]
        ret = {}
        for K in [50]:
            hits_val = self.evaluator.eval(
                {'y_pred_pos': pos_pred, 'y_pred_neg': neg_pred}
            )[f'hits@{K}']
            ret[f'hits{K}'] = hits_val
        return ret

    def _is_better_accuracy(self, loss_all, loss_min_hist, accuracy_all, accuracy_max_hist):
        assert len(loss_all) == len(accuracy_all)
        window_acc = accuracy_all[-self.window_size : ]
        window_loss = loss_all[-self.window_size : ]
        acc_avg = sum(window_acc) / len(window_acc)
        loss_avg = sum(window_loss) / len(window_loss)
        if acc_avg > accuracy_max_hist:
            return True, loss_avg, acc_avg
        else:
            return False, loss_min_hist, accuracy_max_hist

    def _is_better_f1(self, loss_all, loss_min_hist, f1mic_all, f1mic_max_hist, f1mac_all, f1mac_max_hist):
        assert len(loss_all) == len(f1mic_all) == len(f1mac_all)
        window_mic = f1mic_all[-self.window_size : ]
        window_mac = f1mac_all[-self.window_size : ]
        window_loss = loss_all[-self.window_size : ]
        mic_avg = sum(window_mic) / len(window_mic)
        mac_avg = sum(window_mac) / len(window_mac)
        loss_avg = sum(window_loss) / len(window_loss)
        if mic_avg > f1mic_max_hist:
            return True, loss_avg, mic_avg, mac_avg
        else:
            return False, loss_min_hist, f1mic_max_hist, f1mac_max_hist

    def __is_better_hits(self, loss_all, loss_min_hist, hits_all, hits_max_hist):
        assert len(loss_all) == len(hits_all)
        window_hits = hits_all[-self.window_size : ]
        window_loss = loss_all[-self.window_size : ]
        hits_avg = sum(window_hits) / len(window_hits)
        loss_avg = sum(window_loss) / len(window_loss)
        if hits_avg > hits_max_hist:
            return True, loss_avg, hits_avg
        else:
            return False, loss_min_hist, hits_max_hist

    def _is_better_hits20(self, loss_all, loss_min_hist, hits20_all, hits20_max_hist):
        return self.__is_better_hits(loss_all, loss_min_hist, hits20_all, hits20_max_hist)
        
    def _is_better_hits50(self, loss_all, loss_min_hist, hits50_all, hits50_max_hist):
        return self.__is_better_hits(loss_all, loss_min_hist, hits50_all, hits50_max_hist)
    
    def _is_better_hits100(self, loss_all, loss_min_hist, hits100_all, hits100_max_hist):
        return self.__is_better_hits(loss_all, loss_min_hist, hits100_all, hits100_max_hist)