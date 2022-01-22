import os, sys
import torch
from typing import List, Union
from dataclasses import dataclass, field, InitVar
import numpy as np
from torch.autograd import Variable
from shaDow.metric import METRICS, Metrics
from tqdm import tqdm
from graph_engine.frontend import TRAIN, VALID, TEST, MODE2STR, STR2MODE

import shutil
import copy

_bcolors = {
    'header'    : '\033[95m',
    'blue'      : '\033[94m',
    'green'     : '\033[92m',
    'yellow'    : '\033[93m',
    'red'       : '\033[91m',
    'bold'      : '\033[1m',
    'underline' : '\033[4m',
    ''          : '\033[0m',
    None        : ''
}


@dataclass
class InfoBatch:
    """
    Stores all the information of batches within one epoch. 
    """
    batch_size          : List[int] = field(default_factory=list)
    loss                : List[float] = field(default_factory=list)
    labels              : List[np.ndarray] = field(default_factory=list)
    preds               : List[np.ndarray] = field(default_factory=list)
    # book-keeper
    idx_batch           : int=-1
    total_entity        : int=-1        # set at the beginning of each epoch
    PERIOD_LOG          : int=1         # how many batches do we log the info once
    # summary
    names_data_fields = ['batch_size', 'loss', 'labels', 'preds']
    
    def reset(self, total_entity):
        self.idx_batch = -1
        self.total_entity = total_entity
        for n in self.names_data_fields:
            setattr(self, n, [])
    
    @staticmethod
    def _to_numpy(x):
        """
        Convert a PyTorch tensor to numpy array.
        """
        if isinstance(x, Variable):
            x = x.data
        return x.detach().cpu().numpy() if x.is_cuda else x.detach().numpy()

    def add_one_batch(self, idx_batch, info_dict):
        self.idx_batch += 1
        assert self.idx_batch == idx_batch
        if idx_batch % self.PERIOD_LOG == 0:
            for n in self.names_data_fields:
                if isinstance(info_dict[n], Variable):
                    val = self._to_numpy(info_dict[n])
                else:
                    val = info_dict[n]
                getattr(self, n).append(val)


@dataclass
class InfoEpoch:
    """
    Tracks the information of all epochs. Use this class to observe the convergence behavior.
    """
    loss                : List[float] = field(default_factory=list)
    accuracy            : List[float] = None
    f1mic               : List[float] = None
    f1mac               : List[float] = None
    hits20              : List[float] = None
    hits50              : List[float] = None
    hits100             : List[float] = None
    # book-keeper
    idx_epoch           : int=-1
    epoch_best          : int=-1
    loss_min_hist       : float= float('inf')       # min value before the current epoch
    accuracy_max_hist   : float=-float('inf')       # max value before the current epoch
    f1mic_max_hist      : float=-float('inf')       # max value before the current epoch
    f1mac_max_hist      : float=-float('inf')       # max value before the current epoch
    hits20_max_hist     : float=-float('inf')
    hits50_max_hist     : float=-float('inf')
    hits100_max_hist    : float=-float('inf')
    # summary
    names_acc_fields = None
    # Init-fields
    _metric_acc         : InitVar[str]=None
    def __post_init__(self, _metric_acc):
        assert _metric_acc is not None
        self.names_acc_fields = METRICS[_metric_acc]
        for n in self.names_acc_fields:
            setattr(self, n, [])
        self.idx_epoch = -1
        self.epoch_best = -1
    
    def summarize_batches(self, ep, info_batch, f_metric):
        assert ep == self.idx_epoch, "Out of sync between minibatch and logger!!"
        batch_np = np.array(info_batch.batch_size)
        _loss = (np.array(info_batch.loss) * batch_np).sum() / batch_np.sum()
        self.loss.append(_loss)
        y_true = np.concatenate(info_batch.labels)
        y_pred = np.concatenate(info_batch.preds)
        info_metrics = f_metric(y_true, y_pred)
        for k in self.names_acc_fields:
            getattr(self, k).append(info_metrics[k])

    def update_best_metrics(self, *args):
        """args should be the best metric values returned from Metrics class"""
        assert len(args) == len(self.names_acc_fields) + 1      # +1 for loss
        self.loss_min_hist = args[0]        # loss and acc are all averaged over the window
        for i, k in enumerate(self.names_acc_fields):
            setattr(self, f'{k}_max_hist', args[i + 1])
        
    def assert_valid(self, mode, metric_term=None, window_size=None, stochastic_sampler=False):
        l = len(self.loss)
        assert l == self.idx_epoch + 1
        for n in self.names_acc_fields:
            assert l == len(getattr(self, n))
        assert self.idx_epoch >= 0 and self.epoch_best <= self.idx_epoch
        if stochastic_sampler:
            return
        if l > 1 and window_size is not None and mode == VALID and type(metric_term) == tuple:
            str_err = "{}_{}_hist from Metrics returns {}, while from manual sliding window returns {}"
            def construct_win(name):
                _unfold = np.zeros((window_size, l))
                for w in range(window_size):
                    _unfold[w, w:] = getattr(self, name)[w:]
                return _unfold
            assert metric_term[0] in self.names_acc_fields
            n_unfold = construct_win(metric_term[0])
            ep_best_manual = eval(f"n_unfold.mean(axis=0).arg{metric_term[1]}()")
            n_best_manual = n_unfold.mean(axis=0)[ep_best_manual]
            assert n_best_manual == getattr(self, f'{metric_term[0]}_{metric_term[1]}_hist'), \
                str_err.format(
                    metric_term[0], 
                    metric_term[1], 
                    getattr(self, f'{metric_term[0]}_{metric_term[1]}_hist'), n_best_manual
                )
            # check the rest of metrics by ep_best_manual
            loss_min_manual = construct_win('loss').mean(axis=0)[ep_best_manual]
            assert loss_min_manual == self.loss_min_hist, str_err.format(
                'loss', 'min', self.loss_min_hist, loss_min_manual
            )
            for n in self.names_acc_fields:
                n_max_manual = construct_win(n).mean(axis=0)[ep_best_manual]
                assert n_max_manual == getattr(self, f'{n}_max_hist'), str_err.format(
                    n, 'max', getattr(self, f'{n}_max_hist'), n_max_manual
                )


class LoggerBase:
    """
    Base class for logger. Handles printing, saving, loading, logging, summarizing result, etc. 
    """
    style_mode = {
        TRAIN   : None, 
        VALID   : 'green', 
        TEST    : 'yellow'
    }
    style_metric = {
        'loss'  : None,
        'acc'   : 'underline'
    }
    style_status = {
        'running'   : None,
        'final'     : 'bold'
    }
    def __init__(
        self, 
        task: str,
        config_dict: dict, 
        dir_log: str, 
        metric: Metrics, 
        config_term: dict,
        no_log: bool=False, 
        log_test_convergence: int=-1, 
        timestamp: str="",
        period_batch_train: int=1,
        no_pbar: bool=False,
        **kwargs
    ):
        self.task = task
        self.term_window_size = config_term['window_size']
        self.term_window_aggr = config_term['window_aggr']
        self.no_pbar, self.no_log = no_pbar, no_log
        self.dir_log = dir_log
        self.timestamp = timestamp
        self.model_candy = {}           # {ep: model} store the candidate models within the current window
        self.optim_candy = {}           # {ep: optim} store the candidate optimizers within the current window
        self.path_saver = {
            k: f"{dir_log}/saved_{k}_{timestamp}.pkl" for k in ['model', 'optimizer']
        }
        self.path_loader = {'model' : None, "optimizer" : None}
        self.metric = metric
        self.log_test_convergence = log_test_convergence
        assert self.metric.name in [
            "f1", "auc", "accuracy", "accuracy_ogb", 'hits20', 'hits50', 'hits100'
        ]
        self.config_dict = config_dict
        self.file_ep = {m: f"{dir_log}/epoch_{MODE2STR[m]}.csv" for m in [TRAIN, VALID, TEST]}
        self.file_final = f"{dir_log}/final.csv"

        self.info_batch = {
            TRAIN: InfoBatch(PERIOD_LOG=period_batch_train),
            VALID: InfoBatch(PERIOD_LOG=1),
            TEST : InfoBatch(PERIOD_LOG=1)
        }
        self.info_epoch = {
            m: InfoEpoch(_metric_acc=self.metric.name) for m in [TRAIN, VALID, TEST]
        }
        self.acc_final = {}
        self.pbar = None

    def reset(self):
        self.info_batch = {TRAIN    : InfoBatch(PERIOD_LOG=self.info_batch[TRAIN].PERIOD_LOG),
                           VALID    : InfoBatch(PERIOD_LOG=1),
                           TEST     : InfoBatch(PERIOD_LOG=1)}
        self.info_epoch = {m: InfoEpoch(_metric_acc=self.metric.name) for m in [TRAIN, VALID, TEST]}

    def set_loader_path(self, dir_loader):
        """
        Used for inference / re-training. Set the path to load the pre-trained model. 
        """
        assert os.path.isdir(dir_loader), "please provide the dir containing the checkpoints"
        f_pt = [f for f in os.listdir(dir_loader) if f.split('.')[-1] in ['pkl', 'pt']]
        assert 1 <= len(f_pt) <= 2
        for f in f_pt:
            if 'model' in f:
                self.path_loader['model'] = f"{dir_loader}/{f}"
            elif 'optimizer' in f:
                self.path_loader['optimizer'] = f"{dir_loader}/{f}"
            else:
                raise NotImplementedError

    @staticmethod
    def add_logger_args(parser):
        pass

    @staticmethod
    def stringf(msg : Union[str, list], style=None, ending=''):
        """
        Returns formated string so that you can highlight certain text with colors, bold phase etc. 
        """
        if type(style) not in [list, tuple]:
            style = [style]
        _str_style = ''.join(_bcolors[s] for s in style)
        if type(msg) == str:
            subs = f"{_str_style}{msg}{_bcolors['']}"
        elif type(msg) == list:     # list of tuple in the form of [(msg1, style1), (msg2, style2)]
            subs = ''.join(
                f"{_bcolors[se]}{_str_style}{m}{_bcolors['']}" 
                for m, se in msg
            )
        else:
            raise NotImplementedError
        return f"{subs}{ending}"

    @staticmethod
    def printf(msg, style=None):
        print(LoggerBase.stringf(msg, style=style, ending=''))
        
    def update_best_model(self, ep, model, optimizer=None):
        """
        Save the best model so far, flexibly based on the termination criteria. 
        """
        assert len(self.model_candy) <= self.term_window_size
        if len(self.model_candy) == self.term_window_size:
            # cleanup old model first, then store the most recent model
            ep2pop = min(self.model_candy.keys())
            assert ep2pop == ep - self.term_window_size
            del self.model_candy[ep2pop]
            del self.optim_candy[ep2pop]
        self.model_candy[ep] = copy.deepcopy(model).cpu()
        self.optim_candy[ep] = None if optimizer is None else self.model_candy[ep].optimizer
        _info_epoch = self.info_epoch[VALID]
        assert _info_epoch.idx_epoch == ep
        _args = {"loss_all": _info_epoch.loss, "loss_min_hist": _info_epoch.loss_min_hist}
        for n in _info_epoch.names_acc_fields:
            _args[f"{n}_all"] = getattr(_info_epoch, f"{n}")
            _args[f"{n}_max_hist"] = getattr(_info_epoch, f"{n}_max_hist")
        ret_is_better = self.metric.is_better(**_args)
        if ret_is_better[0]:        # flag checking if metric improves
            best_metrics = ret_is_better[1:]
            _info_epoch.update_best_metrics(*best_metrics)
            if self.term_window_aggr == 'center':
                _info_epoch.epoch_best = max(0, ep - self.term_window_size + 1 + self.term_window_size // 2)
            elif self.term_window_aggr.startswith('best_'):
                _mtr_name = self.term_window_aggr[5:]
                window = getattr(_info_epoch, _mtr_name)[-self.term_window_size : ]
                _info_epoch.epoch_best = ep - len(window) + 1 + window.index(max(window))
            elif self.term_window_size == 'last':
                _info_epoch.epoch_best = ep
            assert _info_epoch.epoch_best >= 0, "error in extracting epoch idx from sliding window"
            self.save_model(
                self.model_candy[_info_epoch.epoch_best], 
                optimizer=self.optim_candy[_info_epoch.epoch_best], 
                ep=_info_epoch.epoch_best
            )
            
    def save_model(self, model, optimizer=None, ep=None) -> None:
        self.printf(f"  Saving model {'' if ep is None else ep}...", style="yellow")
        torch.save(model.state_dict(), self.path_saver['model'])
        if optimizer is not None:
            torch.save(optimizer.state_dict(), self.path_saver['optimizer'])
        
    def save_tensor(self, pytensor, fname: str, use_path_loader: bool) -> str:
        self.printf("  Saving tensor ...", style='yellow')
        _path = self.path_saver['model'] if not use_path_loader else self.path_loader['model']
        dir_save = '/'.join(_path.split('/')[:-1])
        fname = fname.format(self.timestamp)
        fname_full = f"{dir_save}/{fname}"
        torch.save(pytensor, fname_full)
        return fname_full

    def restore_model(self, model, optimizer=None, force_reload=False) -> None:
        """
        NOTE: "restore" refers to the loading of model checkpoint of previous epochs.
            To load a model saved in the previous run, call `load_model()` instead. 
        """
        if force_reload or self.info_epoch[VALID].epoch_best >= 0:
            self.printf("  Restoring model ...")
            model.load_state_dict(torch.load(self.path_saver['model']))
            if optimizer is not None:
                optimizer.load_state_dict(torch.load(self.path_saver['optimizer']))
        else:
            self.printf("  NOT restoring model ... PLS CHECK!")
    
    def load_model(self, model, optimizer=None, copy=False, device=None) -> None:
        def gen_new_pt_name(dir_save_new, name):
            file_loaded = self.path_loader[name].split('/')[-1]
            return f"{dir_save_new}/{file_loaded.replace('saved', 'loaded')}"
        self.printf("  Loading model ...")
        dir_save_new = '/'.join(self.path_saver['model'].split('/')[:-1])
        model.load_state_dict(torch.load(self.path_loader['model'], map_location=device))
        file_loaded_model = gen_new_pt_name(dir_save_new, 'model')
        if copy:
            shutil.copyfile(self.path_loader['model'], file_loaded_model)
        else:
            # generate symbol link
            path_rel_model = os.path.relpath(self.path_loader['model'], dir_save_new)
            os.symlink(path_rel_model, file_loaded_model)
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(self.path_loader['optimizer'], map_location=device))
            file_loaded_optm = gen_new_pt_name(dir_save_new, 'optimizer')
            if copy:
                shutil.copyfile(self.path_loader['optimizer'], file_loaded_optm)
            else:
                path_rel_optm = os.path.relpath(self.path_loader['optimizer'], dir_save_new)
                os.symlink(path_rel_optm, file_loaded_optm)


    def epoch_start_reset(self, ep, mode, total_entity):
        self.info_epoch[mode].idx_epoch += 1
        assert self.info_epoch[mode].idx_epoch == ep, "Out of sync of epoch idx between trainer and logger!"
        self.info_batch[mode].reset(total_entity)
        if not self.no_pbar:
            self.pbar = tqdm(total=total_entity, leave=False, file=sys.stdout)
            self.pbar.set_description(
                self.stringf(f"computing {MODE2STR[mode].upper()}", style=self.style_mode[mode])
            )

        
    def init_log2file(self, status='running', meta_info=None):
        if self.no_log:
            return
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)
        if status == 'running':
            self.log2file(TRAIN, "header", meta_info=meta_info, status=status)
            self.log2file(VALID, "header", meta_info=meta_info, status=status)
            if self.log_test_convergence > 0:
                self.log2file(TEST, "header", meta_info=meta_info, status=status)
        elif status == 'final':
            self.log2file("", "header", meta_info=meta_info, status=status)
        else:
            raise NotImplementedError

    def log2file(self, mode, row_type, msg=None, meta_info=None, status='running'):
        """
        mode: train / val / test
        status: running / final
        row_type: header / values
        """
        if row_type == "header":
            assert msg is None
            f_log2file_header = lambda _mode: \
                ', '.join(f"{MODE2STR[_mode]}_{n}" for n in self.info_epoch[_mode].names_acc_fields)
            if status == 'running':
                assert mode in [TRAIN, VALID, TEST]
                _fname = self.file_ep[mode]
                msg = f"epoch, {MODE2STR[mode]}_loss, {f_log2file_header(mode)}\n"
            elif status == "final":
                _fname = self.file_final
                msg = (f"{MODE2STR[TRAIN]}_loss, {f_log2file_header(TRAIN)}, "
                       f"{MODE2STR[VALID]}_loss, {f_log2file_header(VALID)}, "
                       f"{MODE2STR[TEST]}_loss, {f_log2file_header(TEST)}\n")
            else:
                raise NotImplementedError
        elif row_type == "values":
            assert meta_info is None or meta_info != ''
            if status == 'running':
                _fname = self.file_ep[mode]
            elif status == 'final':
                _fname = self.file_final
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if meta_info is not None and meta_info != '':
            msg = f'{meta_info}\n{msg}'
        self._write2file(_fname, msg)
    
    def log_key_step(self, mode, time=-1, status='running'):
        assert mode in [TRAIN, VALID, TEST] and status in ['running', 'final']
        acc_ret = {}
        if status == 'running':
            _info_epoch = self.info_epoch[mode]
            acc_ret['epoch'] = _info_epoch.idx_epoch
            acc_ret['loss']  = _info_epoch.loss[-1]
            epstr = f"Ep {_info_epoch.idx_epoch:4d}" if mode == TRAIN else " " * 7
            msg_print = [(f"{epstr} [{MODE2STR[mode].upper():>6s}]\t", None),
                         (f"loss = {_info_epoch.loss[-1]:.5f}\t", self.style_metric['loss'])]
            for n in _info_epoch.names_acc_fields:
                acc_ret[n] = getattr(_info_epoch, n)[-1]
                msg_print.append((f"{n} = {acc_ret[n]:.5f}\t", self.style_metric['acc']))
            if mode == TRAIN:
                msg_file_header = f'{_info_epoch.idx_epoch:4d}, '
            else:   # reference training epochs
                msg_file_header = f'{_info_epoch.idx_epoch:4d} ({self.info_epoch[TRAIN].idx_epoch:4d}), '
            msg_file_ending = '\n'
        elif status == 'final':
            _info_epoch = self.info_epoch[mode]
            acc_ret['epoch'] = self.info_epoch[VALID].epoch_best
            acc_ret['loss']  = _info_epoch.loss[-1]
            msg_print = [((f"FINAL {MODE2STR[mode].upper()} "
                           f"(Epoch {self.info_epoch[VALID].epoch_best:4d}): \n\t"), None)]
            for n in _info_epoch.names_acc_fields:
                acc_ret[n] = getattr(_info_epoch, n)[-1]
                msg_print.append((f"{n.upper()} = {acc_ret[n]:.5f}\t", self.style_metric['acc']))
            msg_file_header = ''
            msg_file_ending = ', ' if mode != TEST else '\n'
        else:
            raise NotImplementedError
        if time > 0:
            msg_print.append((f"time = {time:.2f} s", None))
        self.printf(msg_print, style=[self.style_mode[mode], self.style_status[status]])
        msg_file = msg_file_header \
                 + f"{_info_epoch.loss[-1]:.5f}, " \
                 + ', '.join(f"{getattr(_info_epoch, n)[-1]:.5f}" for n in _info_epoch.names_acc_fields)\
                 + msg_file_ending
        if not self.no_log:
            self.log2file(mode, "values", msg=msg_file, status=status)
        self.acc_final[mode] = acc_ret
        return acc_ret

    @staticmethod
    def _write2file(filename, logstr, write_mode="a"):
        with open(filename, write_mode) as f:
            f.write(logstr)

    def update_batch(self, mode: str, idx_batch: int, info_batch: dict):
        size_batch = info_batch['batch_size']       # used by tqdm update
        self.info_batch[mode].add_one_batch(idx_batch, info_batch)
        if self.pbar is not None:
            self.pbar.update(size_batch)
    
    def update_epoch(self, ep, mode):
        self.info_epoch[mode].summarize_batches(ep, self.info_batch[mode], self.metric.calc)
        self.info_batch[mode].reset(-1)
        if self.pbar is not None:
            self.pbar.close()

    def validate_result(self, stochastic_sampler: dict):
        for m in [TRAIN, VALID, TEST]:
            self.info_epoch[m].assert_valid(
                m, 
                metric_term=self.metric.metric_term, 
                window_size=self.term_window_size, 
                stochastic_sampler=stochastic_sampler[m]
            )

    def end_training(self, status):
        assert status in ['crashed', 'finished', 'killed']
        if status == 'finished':        # print plain summary: used by wrapper script of ./script/train_multiple_runs.py
            str_summary = "FINAL SUMMARY: "
            for k, v in self.acc_final.items():
                for kk, vv in v.items():
                    str_summary += f'{MODE2STR[k]} {kk} {vv} '
            print(str_summary)
        if self.no_log:
            from itertools import product
            if os.path.exists(self.dir_log):
                assert os.path.isdir(self.dir_log)
                # assert dir_log only contains one *.yml file and one *.pkl / pt file
                f_ymlpt = os.listdir(self.dir_log)
                if len(f_ymlpt) == 1:
                    assert f_ymlpt[0].split('.')[-1] in ['yml', 'yaml'], \
                        f"DIR {self.dir_log} CONTAINS UNKNOWN TYPE OF FILE. ABORTING!"
                else:
                    assert len(f_ymlpt) <= 3
                    assert all(os.path.isfile(f"{self.dir_log}/{f}") for f in f_ymlpt)
                    ext1 = ['yml', 'yaml']
                    ext2 = ['pkl', 'pt']
                    assert set(f.split('.')[-1] for f in f_ymlpt) in [set(p) for p in product(ext1, ext2)], \
                        f"DIR {self.dir_log} CONTAINS UNKNOWN TYPE OF FILE. ABORTING!"
                shutil.rmtree(self.dir_log)
                self.printf(f"Successfully removed log dir {self.dir_log}!", style='red')
        else:
            # move all files from 'running' to the subdir corresponding to status
            dir_split = self.dir_log.split('/')     
            dir_split = dir_split if dir_split[-1] != '' else dir_split[:-1]
            assert dir_split[-2] == 'running'
            dir_split[-2] = status
            dir_new_parent = '/'.join(dir_split[:-1])
            if not os.path.exists(dir_new_parent):
                os.makedirs(dir_new_parent)
            assert os.path.isdir(self.dir_log) and os.path.isdir(dir_new_parent)
            f_logfiles = os.listdir(self.dir_log)
            assert all(os.path.isfile(f"{self.dir_log}/{f}") for f in f_logfiles)
            shutil.move(self.dir_log, dir_new_parent)
            self.printf(f"Successfully moved {self.dir_log} to {dir_new_parent}", style='red')
    
    def decode_csv(self, status, dir_log):
        """
        Used in postproc when we want to match the previously finished runs. 
        """
        assert status == 'final', "Not supporting decoding per epoch files"
        f_csv = f"{dir_log}/final.csv"
        with open(f_csv, 'r') as f:
            lines_record = f.readlines()
        lines_record = [l.strip() for l in lines_record]
        assert len(lines_record) == 2
        keys, values = lines_record
        keys = [k.strip() for k in keys.split(',')]
        values = [float(v.strip()) for v in values.split(',')]
        kv = zip(keys, values)
        ret = {TRAIN: {}, VALID: {}, TEST: {}}
        for k, v in kv:
            m = STR2MODE[k.split('_')[0]]
            ret[m]['_'.join(k.split('_')[1:])] = v
        return ret

    def print_table_postproc(self, acc_orig, acc_post):
        """
        Summarize the acc change in a table after postprocessing. 
        """
        data_line = zip(acc_orig[TRAIN], acc_post[TRAIN], acc_orig[VALID], acc_post[VALID], acc_orig[TEST], acc_post[TEST])
        self.printf(f"==============================================================================", style='bold')
        self.printf(f"TRAIN ORIG -> TRAIN POST    VALID ORIG -> VALID POST    TEST ORIG -> TEST POST", style='bold')
        self.printf(f"------------------------------------------------------------------------------", style=None)
        for dl in data_line:
            self.printf([
                (f"{dl[0]:^10.5f} -> {dl[1]:^10.5f}    ", self.style_mode[TRAIN]),
                (f"{dl[2]:^10.5f} -> {dl[3]:^10.5f}    ", self.style_mode[VALID]),
                (f"{dl[4]:^9.5f} -> {dl[5]:^9.5f}", self.style_mode[TEST])])
        self.printf(f"==============================================================================", style=None)
        acc_orig_copy, acc_post_copy = {}, {}
        for md in [TRAIN, VALID, TEST]:
            acc_orig_copy[md] = np.asarray(acc_orig[md])
            acc_post_copy[md] = np.asarray(acc_post[md])
        self.printf([
            (f"{acc_orig_copy[TRAIN].mean():^10.5f} -> {acc_post_copy[TRAIN].mean():^10.5f}    ", self.style_mode[TRAIN]),
            (f"{acc_orig_copy[VALID].mean():^10.5f} -> {acc_post_copy[VALID].mean():^10.5f}    ", self.style_mode[VALID]),
            (f"{acc_orig_copy[TEST].mean():^9.5f} -> {acc_post_copy[TEST].mean():^9.5f}", self.style_mode[TEST])
        ], style='bold')
        self.printf([
            (f"{acc_orig_copy[TRAIN].std():^10.5f} -> {acc_post_copy[TRAIN].std():^10.5f}    ", self.style_mode[TRAIN]),
            (f"{acc_orig_copy[VALID].std():^10.5f} -> {acc_post_copy[VALID].std():^10.5f}    ", self.style_mode[VALID]),
            (f"{acc_orig_copy[TEST].std():^9.5f} -> {acc_post_copy[TEST].std():^9.5f}", self.style_mode[TEST])
        ], style=None)
        self.printf(f"==============================================================================", style='bold')
