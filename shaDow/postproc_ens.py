# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import shaDow.layers as layers
from shaDow.models import DeepGNN
from shaDow import TRAIN, VALID, TEST
from typing import List, get_type_hints, Union
import itertools



class ModelPostEns(nn.Module):
    def __init__(
                self, 
                dim_in, 
                num_classes, 
                arch_gnn, 
                num_ensemble,
                config_param
            ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hid = arch_gnn['dim']
        self.num_classes = num_classes
        self.num_layers = 1 if 'num_layers' not in arch_gnn else arch_gnn['num_layers']
        self.num_ensemble = num_ensemble
        act = arch_gnn['act']
        if num_ensemble == 1:
            self.ensembler = layers.EnsembleDummy()
        else:       # you may also support other types
            self.ensembler = layers.EnsembleAggregator(
                    self.dim_in, self.dim_hid, num_ensemble, act=act, 
                    dropout=config_param['dropout'], type_dropout=config_param['ensemble_dropout']
                )       # NOTE the output dim is dim_in, not dim_hid
        self.classifier = DeepGNN.NAME2CLS['mlp'](self.dim_in, self.num_classes, act='I', dropout=0.)
        self.sigmoid_loss = arch_gnn['loss'] == 'sigmoid'
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config_param['lr'])

    def _loss(self, preds, labels):
        if self.sigmoid_loss:
            assert preds.shape == labels.shape
            return torch.nn.BCEWithLogitsLoss()(preds, labels) * preds.shape[1]
        else:
            if len(labels.shape) == 2:      # flatten to 1D
                labels = torch.max(labels, axis=1)[1]       # this can handle both bool and float types
            return torch.nn.CrossEntropyLoss()(preds, labels)

    def forward(self, emb_in):
        emb_ens = self.ensembler(emb_in)
        return self.classifier(emb_ens)

    def step(self, mode, status, emb_in, labels):
        assert status in ['running', 'final']
        assert all([e.shape[0] == labels.shape[0] for e in emb_in])
        if len(labels.shape) == 1:
            labels = F.one_hot(labels.to(torch.int64), num_classes=self.num_classes)
        if mode == TRAIN and status == 'running':
            self.train()
            self.optimizer.zero_grad()
            preds = self(emb_in)
            loss = self._loss(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            self.optimizer.step()
        else:
            self.eval()
            with torch.no_grad():
                preds = self(emb_in)
                loss = self._loss(preds, labels)
        assert preds.shape[0] == labels.shape[0]
        return {
            'batch_size': preds.shape[0],
            'loss'      : loss,
            'labels'    : labels,
            'preds'     : self.predict(preds)
        }
    
    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)


class MinibatchPostEns:
    def __init__(self, node_set, batch_size, emb_l, label):
        self.node_set = node_set
        self.batch_size = batch_size
        self.emb_l = emb_l
        self.label = label
        self.num_targets_evaluated = {TRAIN: 0, VALID: 0, TEST: 0}
        self.batch_num = -1

    def to(self, device):
        """
        Mimic the .to() function of torch.nn.Module
        """
        for i in range(len(self.emb_l)):
            if type(self.emb_l[i]) == np.ndarray:
                self.emb_l[i] - torch.from_numpy(self.emb_l[i])
        if type(self.label) == np.ndarray:
            self.label = torch.from_numpy(self.label)
        for k, v in self.node_set.items():
            if type(v) == np.ndarray:
                self.node_set[k] = torch.from_numpy(v)
        self.emb_l = [e.to(device) for e in self.emb_l]
        self.label = self.label.to(device)
        self.node_set = {k: v.to(device) for k, v in self.node_set.items()}
        return self

    def shuffle(self, mode):
        idx = torch.randperm(self.node_set[mode].size(0))
        self.node_set[mode] = self.node_set[mode][idx]
    
    def epoch_start_reset(self, mode):
        self.num_targets_evaluated[mode] = 0
        self.batch_num = -1

    def is_end_epoch(self, mode):
        return self.num_targets_evaluated[mode] >= self.node_set[mode].size(0)

    def one_batch(self, mode):
        self.batch_num += 1
        idx_start = self.num_targets_evaluated[mode]
        idx_end = min(idx_start + self.batch_size, self.node_set[mode].size(0))
        nodes_batch = self.node_set[mode][idx_start : idx_end]
        emb_batch = [e[nodes_batch] for e in self.emb_l]
        label_batch = self.label[nodes_batch]
        self.num_targets_evaluated[mode] = idx_end
        return emb_batch, label_batch



def ensemble(node_set, emb_l, label, config_arch, config_param, logger, device):
    num_ensemble = len(emb_l)
    logger.reset()
    assert all(e.shape == emb_l[0].shape for e in emb_l)
    dim_in = emb_l[0].shape[1]
    num_classes = label.max() + 1 if len(label.shape) == 1 else label.shape[1]
    model = ModelPostEns(dim_in, num_classes, config_arch, num_ensemble, config_param).to(device)
    minibatch = MinibatchPostEns(node_set, config_param['batch_size'], emb_l, label).to(device)
    for ep in range(config_param['end']):
        for md in [TRAIN, VALID, TEST]:
            one_epoch_ens(ep, md, model, minibatch, logger, 'running')
        logger.update_best_model(ep, model, model.optimizer)
    logger.printf(("= = = = = = = = = = = = = = = = = = =\n"
                   "Optimization on [Ensembler] Finished!\n"
                   "= = = = = = = = = = = = = = = = = = =\n"), style="red")
    logger.restore_model(model, optimizer=None)
    acc_ret = {}
    for md in [TRAIN, VALID, TEST]:
        acc_ret[md] = one_epoch_ens(ep + 1, md, model, minibatch, logger, 'final')
    logger.validate_result()
    return {md: a['accuracy'] for md, a in acc_ret.items()}
    

def ensemble_multirun(node_set, emb_pipeline, label, config_arch, config_param, logger, device, acc_record):
    REPEAT_SINGLE_PAIR = 2 if 'repeat_per_emb' not in config_param else config_param['repeat_per_emb']
    emb_pipeline = zip(*list(emb_pipeline.values()))
    acc_ens = []
    for es in emb_pipeline:
        es_flatten = list(itertools.chain.from_iterable(es))
        logger.printf(">>>>>>>>>>>>>>>>>>>", style='red')
        for ir in range(REPEAT_SINGLE_PAIR):
            logger.printf(">>>>>>>>>>>>>>>>>>>", style='red')
            acc_ens.append(ensemble(node_set, es_flatten, label, config_arch, config_param, logger, device))
    return _decode_orig_acc(acc_record)['accuracy'], _merge_stat(acc_ens)


def one_epoch_ens(ep, mode, model, minibatch, logger, status='running'):
    assert status in ['running', 'final']
    assert mode in [TRAIN, VALID, TEST]
    minibatch.epoch_start_reset(mode)
    minibatch.shuffle(mode)
    logger.epoch_start_reset(ep, mode, minibatch.node_set[mode].size(0))
    t1 = time.time()
    while not minibatch.is_end_epoch(mode):
        output_batch = model.step(mode, status, *minibatch.one_batch(mode))
        logger.update_batch(mode, minibatch.batch_num, output_batch)
    t2 = time.time()
    logger.update_epoch(ep, mode)
    return logger.log_key_step(mode, status=status, time=t2 - t1)


def _merge_stat(dict_l : List[dict]):
    key_l = [set(d.keys()) for d in dict_l]
    assert all(k == key_l[0] for k in key_l)
    ret = {k: [] for k in key_l[0]}
    for d in dict_l:
        for k, v in d.items():
            ret[k].append(v)
    return ret

def _decode_orig_acc(dict_sampler : dict):
    acc_to_merge = list(zip(*dict_sampler.values()))
    acc_ret = {k: {TRAIN: [], VALID: [], TEST: []} for k in acc_to_merge[0][0][TRAIN].keys()}
    for acc_pair in acc_to_merge:
        for md in [TRAIN, VALID, TEST]:
            acc_metric = [p[md] for p in acc_pair]
            for k in acc_ret:
                candy = [a[k] for a in acc_metric]
                if k == 'loss':
                    acc_ret[k][md].append(min(candy))
                else:
                    acc_ret[k][md].append(max(candy))
    return acc_ret
