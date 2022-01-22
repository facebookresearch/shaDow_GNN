# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shaDow.layers as layers
from graph_engine.frontend import TRAIN
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Any

from shaDow.minibatch import OneBatchSubgraph


class DeepGNN(nn.Module):
    NAME2CLS = {
        "mlp"   : layers.MLP,
        "gcn"   : layers.GCN,
        "gin"   : layers.GIN,
        "sage"  : layers.GraphSAGE,
        "gat"   : layers.GAT,
        "gatscat": layers.GATScatter,
        "sgc"   : layers.MLPSGC,
        "sign"  : layers.MLPSGC
    }
    def __init__(
        self, 
        dim_feat_raw: int, 
        dim_feat_smooth: int, 
        dim_label_raw: int, 
        dim_label_smooth: int, 
        arch_gnn: Dict[str, Any], 
        aug_feat,
        num_ensemble: int, 
        train_params: Dict[str, Any],
        prediction_task: str
    ):
        """
        Build the multi-layer GNN architecture.

        Inputs:
            num_classes         int, number of classes a node can belong to
            arch_gnn            dict, config for each GNN layer
            train_params        dict, training hyperparameters (e.g., learning rate)
            num_ensemble        int, number of parallel branches to perform subgraph ensemble

        Outputs:
            None
        """
        super().__init__()
        assert prediction_task in {'link', 'node'}, "Only supports node classification and link prediction! "
        self.prediction_task = prediction_task
        self.mulhead = 1
        self.num_gnn_layers = arch_gnn["num_layers"]
        self.num_cls_layers = arch_gnn["num_cls_layers"]
        self.dropout, self.dropedge = train_params["dropout"], train_params['dropedge']
        self.mulhead = int(arch_gnn["heads"])       # only useful for GAT

        self.branch_sharing = arch_gnn['branch_sharing']        # only for ensemble

        self.type_feature_augment = aug_feat
        assert dim_feat_raw <= dim_feat_smooth, "smoothened feature cannot have smaller shape than the original one"
        # NOTE: dim_label_raw may be larger than dim_label_smooth ==> label is not used as input
        self.num_classes = dim_label_raw
        self.dim_label_in = dim_label_smooth
        self.dim_feat_in = dim_feat_smooth
        self.dim_hid = arch_gnn['dim']
        # build the model below
        act, layer_norm = arch_gnn['act'], arch_gnn['layer_norm']
        self.feat_aug_ops = arch_gnn['feature_augment_ops']
        self.aug_layers, self.conv_layers, self.res_pool_layers = [], [], []
        for i in range(num_ensemble):
            # feat aug
            dim_aug_add = 0
            if len(self.type_feature_augment) > 0:
                _dim_aug_out = self.dim_feat_in if self.feat_aug_ops == 'sum' else self.dim_hid
                dim_aug_add += 0 if self.feat_aug_ops == 'sum' else _dim_aug_out
                self.aug_layers.append(
                    nn.ModuleList(
                        nn.Linear(_dim, _dim_aug_out) for _, _dim in self.type_feature_augment
                    )
                )
            # graph convs
            convs = []
            if i == 0 or not self.branch_sharing:
                for j in range(self.num_gnn_layers):
                    dim_in = (self.dim_feat_in + self.dim_label_in + dim_aug_add) if j == 0 else self.dim_hid
                    layer_gconv = DeepGNN.NAME2CLS[arch_gnn['aggr']](
                        dim_in, 
                        self.dim_hid, 
                        dropout=self.dropout, 
                        act=act, 
                        norm=layer_norm, 
                        mulhead=self.mulhead
                    )
                    convs.append(layer_gconv)
                self.conv_layers.append(nn.Sequential(*convs))
            else:       # i > 0 and branch_sharing
                self.conv_layers.append(self.conv_layers[-1])
            # skip-pooling layer
            type_res = arch_gnn['residue'].lower()
            # TODO re-structure yaml config so that pooling params become a dict
            type_pool = arch_gnn['pooling'].split('-')[0].lower()
            args_pool = {}
            if type_pool == 'sort':
                args_pool['k'] = int(arch_gnn['pooling'].split('-')[1])
            layer_res_pool = layers.ResPool(
                self.dim_hid, 
                self.dim_hid, 
                self.num_gnn_layers, 
                type_res, 
                type_pool, 
                dropout=self.dropout,
                act=act, 
                args_pool=args_pool, 
                prediction_task=self.prediction_task
            )
            self.res_pool_layers.append(layer_res_pool)
        if len(self.aug_layers) > 0:
            self.aug_layers = nn.ModuleList(self.aug_layers)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.res_pool_layers = nn.ModuleList(self.res_pool_layers)
        # ------- ensembler + classifier -------
        if num_ensemble == 1:
            self.ensembler = layers.EnsembleDummy()
        else:
            self.ensembler = layers.EnsembleAggregator(
                self.dim_hid, 
                self.dim_hid, 
                num_ensemble, 
                dropout=self.dropout, 
                type_dropout=train_params["ensemble_dropout"], 
                act=arch_gnn["ensemble_act"]
            )
        _norm_type = 'norm_feat' if self.prediction_task == 'node' else 'none'
        # (multi-layer) classifier: by default, we set number of MLP classifier layers to be 1
        self.classifier = []
        for i in range(self.num_cls_layers):
            if i < self.num_cls_layers - 1:
                _kwargs = {'dim_out': self.dim_hid, 'act': act, 'dropout': self.dropout}
            else:
                _kwargs = {'dim_out': self.num_classes, 'act': 'I', 'dropout': 0.}
            _kwargs.update({'dim_in': self.dim_hid, 'norm': _norm_type})
            self.classifier.append(DeepGNN.NAME2CLS['mlp'](**_kwargs))
        self.classifier = nn.Sequential(*self.classifier)
        # ---- optimizer, etc. ----
        self.lr = train_params["lr"]
        self.sigmoid_loss = arch_gnn["loss"] == "sigmoid"
        self.loss, self.opt_op = 0, None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.num_ensemble = num_ensemble


    def _loss(self, preds, labels):
        """
        The predictor performs sigmoid (for multi-class) or softmax (for single-class)
        """
        if self.sigmoid_loss:
            assert preds.shape == labels.shape
            return torch.nn.BCEWithLogitsLoss()(preds, labels.type(preds.dtype)) * preds.shape[1]
        else:
            if len(labels.shape) == 2:      # flatten to 1D
                labels = torch.max(labels, axis=1)[1]       # this can handle both bool and float types
            return torch.nn.CrossEntropyLoss()(preds, labels)


    def forward(
        self, 
        mode, 
        feat_ens, 
        adj_ens, 
        target_ens, 
        size_subg_ens, 
        feat_aug_ens, 
        dropedge
    ):
        num_ensemble = len(feat_ens)
        emb_subg_ens = []
        for i in range(num_ensemble):
            if self.dim_label_in > 0 and mode == TRAIN:
                feat_ens[i][target_ens[i], -self.dim_label_in:] = 0
            # feature augment
            if len(self.type_feature_augment) > 0:
                for ia, (ta, _dim) in enumerate(self.type_feature_augment):
                    feat_aug_emb = self.aug_layers[i][ia](feat_aug_ens[i][ta])
                    if self.feat_aug_ops == 'sum':
                        feat_ens[i][:, :self.dim_feat_in] += feat_aug_emb
                    else:
                        feat_ens[i] = torch.cat([feat_ens[i], feat_aug_emb], dim=1).to(feat_ens[i].device)
            # main propagation
            xjk = []
            xmd = (feat_ens[i], adj_ens[i], False, dropedge)
            for md in self.conv_layers[i]:
                xmd = md(xmd, sizes_subg=size_subg_ens[i])
                xjk.append(xmd[0])
            # residue and pooling
            emb_subg_i = self.res_pool_layers[i](xjk, target_ens[i], size_subg_ens[i])
            emb_subg_i = F.normalize(emb_subg_i, p=2, dim=1)
            emb_subg_ens.append(emb_subg_i)
        emb_ensemble = self.ensembler(emb_subg_ens)
        pred_subg = self.classifier(emb_ensemble)
        return pred_subg, emb_subg_ens

    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)

    def step(self, mode, status, batch_data: OneBatchSubgraph):
        assert status in ['running', 'final']
        args_forward_common = batch_data.to_dict(
            {"feat_ens", "adj_ens", "target_ens", "size_subg_ens", "feat_aug_ens"}
        )
        label_targets = batch_data.label
        if len(label_targets.shape) == 1 and self.num_classes > 1:
            label_targets = F.one_hot(label_targets.to(torch.int64), num_classes=self.num_classes)
        if mode == TRAIN and status == 'running':
            self.train()
            self.optimizer.zero_grad()
            preds, emb_ens = self(mode, dropedge=self.dropedge, **args_forward_common)
            loss = self._loss(preds, label_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            self.optimizer.step()
        else:
            self.eval()
            with torch.no_grad():
                preds, emb_ens = self(mode, dropedge=0., **args_forward_common)
                loss = self._loss(preds, label_targets)
        assert preds.shape[0] == label_targets.shape[0]
        return {
            'batch_size': preds.shape[0],
            'loss'      : loss,
            'labels'    : label_targets,
            'preds'     : self.predict(preds),
            'emb_ens'   : emb_ens
        }

    def calc_complexity_step(self, adj_ens, feat_ens, sizes_subg_ens):
        """
        The complexity of generating the prediction. 
        """
        num_ensemble = len(feat_ens)
        dims_ens = []
        ops = 0
        for i in range(num_ensemble):
            if len(self.type_feature_augment) > 0:
                for ia, _ in enumerate(self.type_feature_augment): # this does not change feat dim
                    ops += np.prod(list(self.aug_layers[i][ia].weight.shape)) * feat_ens[i].shape[0]
            dims_respool = []
            dims_conv = (
                layers.Dims_X(*(feat_ens[i].shape)), 
                layers.Dims_adj(adj_ens[i].shape[0], adj_ens[i].size)
            )
            for md in self.conv_layers[i]:
                dims_conv, _ops = md.complexity(*dims_conv)
                dims_respool.append(dims_conv[0])
                ops += _ops
            dims_emb, _ops = self.res_pool_layers[i].complexity(dims_respool, sizes_subg_ens)
            ops += _ops
            dims_ens.append(dims_emb)
        dims_cls, _ops = self.ensembler.complexity(dims_ens)
        ops += _ops
        _dims_final, _ops = self.classifier.complexity(dims_cls)
        ops += _ops
        return ops

    def __str__(self):
        instance_info = f"model name: {type(self).__name__}"
        return instance_info
