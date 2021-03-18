# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shaDow.layers as layers
from shaDow import TRAIN, VALID, TEST
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DeepGNN(nn.Module):
    NAME2CLS = {"mlp"   : layers.MLP,
                "gcn"   : layers.GCN,
                "gin"   : layers.GIN,
                "sage"  : layers.GraphSAGE,
                "gat"   : layers.GAT,
                "gatscat": layers.GATScatter,
                "sgc"   : layers.MLPSGC,
                "sign"  : layers.MLPSGC}
    def __init__(
                self, 
                dim_feat_raw, 
                dim_feat_smooth, 
                dim_label_raw, 
                dim_label_smooth, 
                arch_gnn, 
                aug_feat,
                num_ensemble, 
                train_params
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
        self.mulhead = 1
        self.num_layers = arch_gnn["num_layers"]
        self.dropout, self.dropedge = train_params["dropout"], train_params['dropedge']
        self.mulhead = int(arch_gnn["heads"])       # only useful for GAT

        self.branch_sharing = arch_gnn['branch_sharing']        # only for ensemble

        self.type_feature_augment = aug_feat
        assert dim_feat_raw <= dim_feat_smooth, "smoothened feature cannot have smaller shape than the original one"
        # NOTE: dim_label_raw may be larger than dim_label_smooth ==> label is not used as input
        self.num_classes = dim_label_raw
        self.dim_label_in = dim_label_smooth
        self.dim_feat_in = dim_feat_smooth
        self.dim_hidden = arch_gnn['dim']
        # build the model below
        dim, act = arch_gnn['dim'], arch_gnn['act']
        self.aug_layers, self.conv_layers, self.res_pool_layers = [], [], []
        for i in range(num_ensemble):
            # feat aug
            if len(self.type_feature_augment) > 0:
                self.aug_layers.append(nn.ModuleList(
                    nn.Linear(_dim, self.dim_feat_in) for _, _dim in self.type_feature_augment
                ))
            # graph convs
            convs = []
            if i == 0 or not self.branch_sharing:
                for j in range(arch_gnn['num_layers']):
                    cls_gconv = DeepGNN.NAME2CLS[arch_gnn['aggr']]
                    dim_in = (self.dim_feat_in + self.dim_label_in) if j == 0 else dim
                    convs.append(cls_gconv(dim_in, dim, dropout=self.dropout, act=act, mulhead=self.mulhead))
                self.conv_layers.append(nn.Sequential(*convs))
            else:       # i > 0 and branch_sharing
                self.conv_layers.append(self.conv_layers[-1])
            # skip-pooling layer
            type_res = arch_gnn['residue'].lower()
            type_pool = arch_gnn['pooling'].split('-')[0].lower()
            cls_res_pool = layers.ResPool
            args_pool = {}
            if type_pool == 'sort':
                args_pool['k'] = int(arch_gnn['pooling'].split('-')[1])
            self.res_pool_layers.append(
                cls_res_pool(dim, dim, arch_gnn['num_layers'], type_res, type_pool,
                    dropout=self.dropout, act=act, args_pool=args_pool
                ))
        if len(self.aug_layers) > 0:
            self.aug_layers = nn.ModuleList(self.aug_layers)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.res_pool_layers = nn.ModuleList(self.res_pool_layers)
        # ------- ensembler + classifier -------
        if num_ensemble == 1:
            self.ensembler = layers.EnsembleDummy()
        else:
            self.ensembler = layers.EnsembleAggregator(dim, dim, num_ensemble, dropout=self.dropout, 
                        type_dropout=train_params["ensemble_dropout"], act=arch_gnn["ensemble_act"])
        self.classifier = DeepGNN.NAME2CLS['mlp'](dim, self.num_classes, act='I', dropout=0.)
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
            return torch.nn.BCEWithLogitsLoss()(preds, labels) * preds.shape[1]
        else:
            if len(labels.shape) == 2:      # flatten to 1D
                labels = torch.max(labels, axis=1)[1]       # this can handle both bool and float types
            return torch.nn.CrossEntropyLoss()(preds, labels)


    def forward(self, mode, feat_ens, adj_ens, target_ens, 
                size_subg_ens, feat_aug_ens, dropedge):
        num_ensemble = len(feat_ens)
        emb_subg_ens = []
        for i in range(num_ensemble):
            if self.dim_label_in > 0 and mode == TRAIN:     # TODO: mask out valid nodes to better terminate
                feat_ens[i][target_ens[i], -self.dim_label_in:] = 0
            # feature augment
            if len(self.type_feature_augment) > 0:
                for ia, (ta, _dim) in enumerate(self.type_feature_augment):
                    feat_ens[i][:, :self.dim_feat_in] += self.aug_layers[i][ia](feat_aug_ens[i][ta])
            assert self.dim_label_in + self.dim_feat_in == feat_ens[i].shape[1]
            # main propagation
            xjk = []
            xmd = (feat_ens[i], adj_ens[i], False, dropedge)
            for md in self.conv_layers[i]:
                xmd = md(xmd)
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


    def step(self, mode, status, adj_ens, feat_ens, label_ens, 
             size_subg_ens, target_ens, feat_aug_ens=None):
        assert status in ['running', 'final']
        args_forward_common = {
            "feat_ens"  : feat_ens,
            "adj_ens"   : adj_ens,
            "target_ens": target_ens,
            "size_subg_ens" : size_subg_ens,
            "feat_aug_ens"  : feat_aug_ens
        }
        label_targets = label_ens[0][target_ens[0]]
        if len(label_targets.shape) == 1:
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
        return {'batch_size': preds.shape[0],
                'loss'      : loss,
                'labels'    : label_targets,
                'preds'     : self.predict(preds),
                'emb_ens'   : emb_ens}

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
            dims_conv = (layers.Dims_X(*(feat_ens[i].shape)), layers.Dims_adj(adj_ens[i].shape[0], adj_ens[i].size))
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
