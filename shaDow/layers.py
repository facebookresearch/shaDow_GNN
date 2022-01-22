import torch
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F
from shaDow.utils import idx_nD_list, set_nD_list, construct_nD_list
from graph_engine.frontend.graph_utils import (
    coo_scipy2torch, get_deg_torch_sparse, adj_norm_sym, adj_norm_rw
)
from torch_scatter import scatter
from torch_geometric.nn import global_sort_pool
import numpy as np
import torch.nn.functional as F
from shaDow.utils import cartesian_product

from collections import namedtuple

Dims_X = namedtuple('Dims_X', ['num_nodes', 'num_feats'])
Dims_adj = namedtuple('Dims_adj', ['num_nodes', 'num_edges'])

# TODO: can make F_ACT a @dataclass

F_ACT = {
    "relu"     : (nn.ReLU, {}),
    "I"        : (nn.LeakyReLU, {"negative_slope": (lambda kwargs: 1)}),
    "elu"      : (nn.ELU, {}),
    "tanh"     : (nn.Tanh, {}),
    "leakyrelu": (nn.LeakyReLU, {"negative_slope": (lambda kwargs: 0.2)}),
    "prelu"    : (nn.PReLU, {}),
    "prelu+"   : (nn.PReLU, {"num_parameters": (lambda kwargs: kwargs['dim_out'])})
}


def get_torch_act(act, args):
    _torch_args = {k: v(args) for k, v in F_ACT[act][1].items()}
    return F_ACT[act][0](**_torch_args)


class EnsembleDummy(nn.Module):
    """ used when there is only one branch of subgraph (i.e., no ensemble) """
    def __init__(self, dim_in=0, dim_out=0, **kwargs):
        super().__init__()

    def forward(self, Xi):
        assert len(Xi) == 1, "ONLY USE DUMMY ENSEMBLER WITH ONE BRANCH!"
        return Xi[0]
    
    def complexity(self, dims):
        assert len(dims) == 1, "ONLY USE DUMMY ENSEMBLER WITH ONE BRANCH!"
        return dims[0], 0



class ResPool(nn.Module):
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        num_layers: int, 
        type_res: str, 
        type_pool: str, 
        dropout: float, 
        act: str, 
        args_pool: dict=None, 
        prediction_task: str='node'
    ):
        super().__init__()
        self.dim_out = dim_out
        self.type_pool = type_pool
        self.type_res = type_res
        self.prediction_task = prediction_task
        if type_pool == 'center':
            if type_res == 'none':
                if self.prediction_task == 'node':
                    self.dim_in = self.dim_out = 0
                else:
                    self.dim_in = dim_in
            elif type_res in ['cat', 'concat']:
                # This is equivalent to regular JK
                # 1. take center node for each feat-\ell
                # 2. concatenate multi-scale feat of center nodes
                # 3. MLP
                self.dim_in = num_layers * dim_in
            else:       # replace step 2 with max / mean
                self.dim_in = dim_in
        else:       # e.g., sort / max / mean / sum
            # 1. pool all # layer feats
            # 2. cat center node for each layer
            # 3. cat outputs of 1. and 2., feed to MLP
            if type_res in ['cat', 'concat']:
                self.dim_in = 2 * dim_in * num_layers       # MLP dimension after pooling
            else:
                self.dim_in = 2 * dim_in
            if type_pool == 'sort':
                # [pool input]        -> [pool MLP input]    -> [pool MLP output]
                # N * self.dim_in / 2 -> k * self.dim_in / 2 -> self.dim_in / 2
                assert 'k' in args_pool, "Sort pooling needs the budget k as input!"
                self.k = args_pool['k']
                _f_lin_pool = nn.Linear(self.k * int(self.dim_in / 2), int(self.dim_in / 2))
                _f_dropout_pool = nn.Dropout(p=dropout)
                _act_pool = get_torch_act(act, locals())
                self.nn_pool = nn.Sequential(_f_dropout_pool, _f_lin_pool, _act_pool)
        if self.dim_in > 0 and self.dim_out > 0:
            _act = get_torch_act(act, locals())
            _f_lin = nn.Linear(self.dim_in, self.dim_out, bias=True)
            _f_dropout = nn.Dropout(p=dropout)
            self.nn = nn.Sequential(_f_dropout, _f_lin, _act)
            self.offset = nn.Parameter(torch.zeros(self.dim_out))
            self.scale = nn.Parameter(torch.ones(self.dim_out))

    def f_norm(self, _feat):
        mean = _feat.mean(dim=1).view(_feat.shape[0], 1)
        var = _feat.var(dim=1, unbiased=False).view(_feat.shape[0], 1) + 1e-9
        feat_out = (_feat - mean) * self.scale * torch.rsqrt(var) + self.offset
        return feat_out
    
    def f_residue(self, feat_l):
        # 'none' residue is handled separately in forward()
        if self.type_res in ['cat', 'concat']:
            feat_ret = torch.cat(feat_l, dim=1)
        elif self.type_res == 'sum':
            feat_ret = torch.stack(feat_l, dim=0).sum(dim=0)
        elif self.type_res == 'max':
            feat_ret = torch.max(torch.stack(feat_l, dim=0), dim=0).values
        else:
            raise NotImplementedError
        return feat_ret
    
    def f_res_complexity(self, dim_x_l):
        """
        returns feature dim after residue, ops due to residue aggregation
        """
        if self.type_res in ['cat', 'concat']:
            return sum([d.num_feats for d in dim_x_l]), 0
        elif self.type_res == 'sum':
            return dim_x_l[-1].num_feats, (len(dim_x_l) - 1) * (dim_x_l[-1].num_nodes * dim_x_l[-1].num_feats)
        elif self.type_res == 'max':
            return dim_x_l[-1].num_feats, (len(dim_x_l) - 1) * (dim_x_l[-1].num_nodes * dim_x_l[-1].num_feats)
        else:
            raise NotImplementedError

    def aggr_target_emb(self, feat_src_dst):
        if self.prediction_task == 'node':
            return feat_src_dst
        else:
            assert len(feat_src_dst.shape) == 2
            b, f = feat_src_dst.shape
            feat_ret = feat_src_dst.reshape(b//2, 2, f)
            return feat_ret[:, 0] * feat_ret[:, 1]

    def forward(self, feats_in_l, idx_targets, sizes_subg):
        if self.prediction_task == 'link':
            assert idx_targets.shape[0] == 2 * sizes_subg.shape[0]
        else:
            assert idx_targets.shape[0] == sizes_subg.shape[0]
        if self.type_pool == 'center':
            if self.type_res == 'none':
                feat_in = feats_in_l[-1][idx_targets]
                if self.prediction_task == 'node':
                    return feat_in
            else:       # regular JK
                feats_root_l = [f[idx_targets] for f in feats_in_l]
                feat_in = self.f_residue(feats_root_l)
            feat_in = self.aggr_target_emb(feat_in)
        elif self.type_pool in ['max', 'mean', 'sum']:
            # first pool subgraph at each layer, then residue
            offsets = torch.cumsum(sizes_subg, dim=0)
            offsets = torch.roll(offsets, 1)
            offsets[0] = 0
            idx = torch.arange(feats_in_l[-1].shape[0]).to(feats_in_l[-1].device)
            if self.type_res == 'none':
                feat_pool = F.embedding_bag(idx, feats_in_l[-1], offsets, mode=self.type_pool)
                feat_root = feats_in_l[-1][idx_targets]
            else:
                feat_pool_l = []
                for feat in feats_in_l:
                    feat_pool = F.embedding_bag(idx, feat, offsets, mode=self.type_pool)
                    feat_pool_l.append(feat_pool)
                feat_pool = self.f_residue(feat_pool_l)
                feat_root = self.f_residue([f[idx_targets] for f in feats_in_l])
            feat_in = torch.cat([self.aggr_target_emb(feat_root), feat_pool], dim=1)
        elif self.type_pool == 'sort':
            if self.type_res == 'none':
                feat_pool_in = feats_in_l[-1]
                feat_root = feats_in_l[-1][idx_targets]
            else:
                feat_pool_in = self.f_residue(feats_in_l)
                feat_root = self.f_residue([f[idx_targets] for f in feats_in_l])
            arange = torch.arange(sizes_subg.size(0)).to(sizes_subg.device)
            idx_batch = torch.repeat_interleave(arange, sizes_subg)
            feat_pool_k = global_sort_pool(feat_pool_in, idx_batch, self.k)     # #subg x (k * F)
            feat_pool = self.nn_pool(feat_pool_k)
            feat_in = torch.cat([self.aggr_target_emb(feat_root), feat_pool], dim=1)
        else:
            raise NotImplementedError
        return self.f_norm(self.nn(feat_in))

    def complexity(self, dims_x_l, sizes_subg):
        num_nodes = len(sizes_subg)
        num_neigh = dims_x_l[-1].num_nodes
        assert num_neigh == sizes_subg.sum()
        if self.type_pool == 'center':
            if self.type_res == 'none':
                return Dims_X(num_nodes, dims_x_l[-1].num_feats), 0
            else:       # regular JK
                dims_root_l = [Dims_X(num_nodes, d.num_feats) for d in dims_x_l]
                dim_f, ops = self.f_res_complexity(dims_root_l)    # pool first, then residue
                return Dims_X(num_nodes, dim_f), ops
        elif self.type_pool in ['max', 'mean', 'sum']:
            ops = dims_x_l[-1].num_nodes * dims_x_l[-1].num_feats
            mult = 1 if self.type_res == 'none' else len(dims_x_l)
            ops *= mult     # we first pool graph
            dims_root_l = [Dims_X(num_nodes, d.num_feats) for d in dims_x_l]
            _dim_f, ops_res = self.f_res_complexity(dims_root_l)
            ops += 2 * ops_res      # "2" since one for neighs, and the other for root
        elif self.type_pool == 'sort':
            if self.type_res == 'none':
                ops = 0
            else:
                _dim, ops = self.f_res_complexity(dims_x_l)
            # global_sort_pool: sort is only alone last channel, therefore neglect its complexity
            for n in self.nn_pool:
                if type(n) == nn.Linear:
                    ops += np.pool(list(n.weight.shape)) * num_nodes
            ops += np.prod(list(self.nn_pool.weight.shape)) * self.k * num_nodes
        for n in self.nn:
            if type(n) == nn.Linear:
                ops += np.prod(list(n.weight.shape)) * num_nodes
                dim_f = n.weight.shape[0]       # dim 0 is output dim
        return Dims_X(num_nodes, dim_f), ops


class EnsembleAggregator(nn.Module):
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        num_ensemble: int, 
        dropout: float=0.0, 
        act: str="leakyrelu", 
        type_dropout: str="none"
    ):
        r"""
        Embedding matrix from branches: X_i
        Output matrix after ensemble: Y

        Learnable parameters (shared across branches): 
        * W \in R^{dim_in x dim_out}
        * b \in R^{dim_out}
        * q \in R^{dim_out}

        Operations:
        1. w_i = act(X_i W + b) q
        2. softmax along the i dimension
        3. Y = \sum w_i X_i
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.act = nn.ModuleList(get_torch_act(act, locals()) for _ in range(num_ensemble))
        self.f_lin = nn.Linear(dim_in, dim_out, bias=True)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.q = nn.Parameter(torch.ones(dim_out))
        assert type_dropout in ["none", "feat", "coef"]
        self.type_dropout = type_dropout

    def forward(self, Xi):
        omega_ensemble = []
        for i, X in enumerate(Xi):
            if self.type_dropout == "none":
                X_ = X
            elif self.type_dropout == "coef":
                X_ = self.f_dropout(X)
            else:
                Xi[i] = self.f_dropout(X)
                X_ = Xi[i]
            omega_ensemble.append(self.act[i](self.f_lin(X_)).mm(self.q.view(-1, 1)))
        omega_ensemble = torch.cat(omega_ensemble, 1)
        omega_norm = F.softmax(omega_ensemble, dim=1)
        Y = 0
        for i, X in enumerate(Xi):
            Y += omega_norm[:, i].view(-1, 1) * X
        return Y

    def complexity(self, dims_x_l):
        ops = 0
        for dx in dims_x_l:
            assert dx.num_feats == self.f_lin.weight.shape[1]
            ops += dx.num_nodes * dx.num_feats * self.f_lin.weight.shape[0]    # X W
            ops += dx.num_nodes * dx.num_feats      # X q
            ops += dx.num_nodes * dx.num_feats      # sum X[i]
        return Dims_X(dx.num_nodes, self.f_lin.weight.shape[0]), ops
            

class shaDowLayer(nn.Module):
    """
    Parent class for various GNN backbone message passing
    e.g., GCN, SAGE, GIN, GAT, etc. 
    """
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        dropout: float=0.0, 
        act: str='relu', 
        norm: str='norm_feat', 
        **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.dim_in, self.dim_out = dim_in, dim_out
        self.act = get_torch_act(act, locals())
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.norm = norm
        self.norm_dim = (1, dim_out) if 'norm_dim' not in kwargs else kwargs['norm_dim']
        if norm == 'norm_feat':     # we always use this type of norm in our paper
            self.offset = nn.Parameter(torch.zeros(self.norm_dim))
            self.scale = nn.Parameter(torch.ones(self.norm_dim))
        elif norm == 'pairnorm':
            self.pairnorm_s = 1 if 'pairnorm_s' not in kwargs else kwargs['pairnorm_s']
            
    def spmm(self, adj, X):
        return torch.sparse.mm(adj, X)

    def _f_norm_feat(self, feat_in, feat_out, loop_idx):
        # iterate the nested indices with a single for loop
        for inp in loop_idx:
            i = tuple(inp)      # need to index numpy array with tuple
            _feat = idx_nD_list(feat_in, i)
            mean = _feat.mean(dim=1, keepdim=True)
            var = _feat.var(dim=1, unbiased=False).view(_feat.shape[0], 1) + 1e-9
            feat_norm = (_feat - mean) * self.scale[i] * torch.rsqrt(var) + self.offset[i]
            set_nD_list(feat_out, i, feat_norm)
        return feat_out
    
    def _f_pairnorm(self, feat_in, feat_out, loop_idx, sizes_subg):
        subg_size_offsets = torch.cumsum(sizes_subg, dim=0)
        subg_size_offsets = torch.roll(subg_size_offsets, 1)
        subg_size_offsets[0] = 0
        idx_bag = None
        for inp in loop_idx:
            i = tuple(inp)
            # the first dim of feat_in should all be the same
            feat = idx_nD_list(feat_in, i)
            if idx_bag is None:
                idx_bag = torch.arange(feat.shape[0]).to(feat.device)
            else:
                assert idx_bag.size(0) == feat.shape[0]
            feat_mean = F.embedding_bag(idx_bag, feat, subg_size_offsets, mode='mean')
            feat_mean = feat_mean.repeat_interleave(sizes_subg, dim=0)
            feat_centered = feat - feat_mean
            idx_null = torch.where(feat_centered.sum(dim=1) == 0)
            feat_centered[idx_null] = feat[idx_null]
            breakpoint()        # TODO: bug here. should have 1/n = 1/|V_sub| factor
            feat_norm = feat_centered / ((feat_centered**2).sum(dim=1, keepdim=True))**0.5
            set_nD_list(feat_out, i, feat_norm)
        return feat_out

    def f_norm(self, feat_l, sizes_subg):
        if self.norm == 'none':
            return feat_l
        loop_idx = cartesian_product(*[np.arange(d) for d in self.norm_dim[:-1]])   # last dim is feature dim
        feat_in = feat_l
        # init empty feat_out
        feat_out = construct_nD_list(self.norm_dim[:-1])
        if self.norm == 'norm_feat':
            return self._f_norm_feat(feat_in, feat_out, loop_idx)
        elif self.norm == 'pairnorm':
            return self._f_pairnorm(feat_in, feat_out, loop_idx, sizes_subg)


class MLP(shaDowLayer):
    def __init__(self, dim_in, dim_out, dropout=0.0, act="relu", norm='norm_feat', **kwargs):
        assert norm in ['norm_feat', 'none']    # assert MLP norm is not pairnorm
        kwargs['norm_dim'] = (1, dim_out)
        super().__init__(dim_in, dim_out, dropout=dropout, act=act, norm=norm, **kwargs)
        self.f_lin = nn.Linear(dim_in, dim_out)

    def forward(self, feat_in):
        """
        Inputs:
            feat_in         2D matrix of input node features

        Outputs:
            feat_out        2D matrix of output node features
        """
        # dropout-act-norm
        feat_in = self.f_dropout(feat_in)
        feat_out = self.act(self.f_lin(feat_in))
        feat_out = self.f_norm([feat_out], None)[0]
        return feat_out

    def complexity(self, dims_x):
        assert dims_x.num_feats == self.f_lin.weight.shape[1]
        ops = dims_x.num_nodes * np.product(self.f_lin.weight.shape)
        return Dims_X(dims_x.num_nodes, self.f_lin.weight.shape[0]), ops


class MLPSGC(MLP):
    """
    MLP layer used in SGC arch. Only differs from the parent class of MLP by the input args to `forward()`
    """
    def __init__(self, dim_in, dim_out, dropout=0.0, act="relu", norm='norm_feat', **kwargs):
        super().__init__(dim_in, dim_out, dropout=dropout, act=act, norm=norm, **kwargs)
    
    def forward(self, inputs, **kwargs):
        assert type(inputs) in [list, tuple] and len(inputs) == 4        # feat_in, adj, is_normed, dropedge
        feat_in = inputs[0]
        feat_out = super().forward(feat_in)
        return feat_out, None, None, None


class GCN(shaDowLayer):
    def __init__(self, dim_in, dim_out, dropout=0.0, act="relu", norm='norm_feat', **kwargs):
        kwargs['norm_dim'] = (1, dim_out)
        super().__init__(dim_in, dim_out, dropout=dropout, act=act, norm=norm, **kwargs)
        self.f_lin = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, inputs, sizes_subg):
        feat_in, adj, is_normed, dropedge = inputs
        feat_in = self.f_dropout(feat_in)
        if not is_normed and adj is not None:
            assert type(adj) == sp.csr_matrix
            adj_norm = adj_norm_sym(adj, dropedge=dropedge)     # self-edges are already added by C++ sampler
            adj_norm = coo_scipy2torch(adj_norm.tocoo()).to(feat_in.device)
        else:
            assert adj is None or type(adj) == torch.Tensor
            adj_norm = adj
        feat_aggr = torch.sparse.mm(adj_norm, feat_in)
        feat_trans = self.f_lin(feat_aggr)
        feat_out = self.f_norm([self.act(feat_trans)], sizes_subg)[0]
        return feat_out, adj_norm, True, 0.

    def complexity(self, dims_x, dims_adj):
        ops = dims_adj.num_edges * dims_x.num_feats \
            + dims_x.num_nodes * np.product(self.f_lin.weight.shape)
        return (
            Dims_X(dims_x.num_nodes, self.f_lin.weight.shape[0]),
            Dims_adj(dims_adj.num_nodes, dims_adj.num_edges)
        ), ops


class GraphSAGE(shaDowLayer):
    def __init__(self, dim_in, dim_out, dropout=0.0, act="relu", norm='norm_feat', **kwargs):
        kwargs['norm_dim'] = (2, dim_out)   # 2 for self + neigh
        super().__init__(dim_in, dim_out, dropout=dropout, act=act, norm=norm, **kwargs)
        self.f_lin_self = nn.Linear(dim_in, dim_out)
        self.f_lin_neigh = nn.Linear(dim_in, dim_out)

    def forward(self, inputs, sizes_subg):
        """
        Inputs:
            adj_norm        normalized adj matrix of the subgraph
            feat_in         2D matrix of input node features

        Outputs:
            adj_norm        same as input (to facilitate nn.Sequential)
            feat_out        2D matrix of output node features
        """
        # dropout-act-norm
        feat_in, adj, is_normed, dropedge = inputs
        if not is_normed and adj is not None:
            assert type(adj) == sp.csr_matrix
            adj = coo_scipy2torch(adj.tocoo()).to(feat_in.device)
            adj_norm = adj_norm_rw(adj, dropedge=dropedge)
        else:
            assert adj is None or type(adj) == torch.Tensor or type(adj) == tuple
            adj_norm = adj
        feat_in = self.f_dropout(feat_in)
        feat_self = feat_in
        feat_neigh = self.spmm(adj_norm, feat_in)
        feat_self_trans, feat_neigh_trans = self.f_norm(
            [
                self.act(self.f_lin_self(feat_self)),
                self.act(self.f_lin_neigh(feat_neigh))
            ], 
            sizes_subg
        )
        feat_out = feat_self_trans + feat_neigh_trans
        return feat_out, adj_norm, True, 0.

    def complexity(self, dims_x, dims_adj):
        assert dims_x.num_nodes == dims_adj.num_nodes
        ops = dims_x.num_nodes * np.product(self.f_lin_self.weight.shape) \
            + dims_adj.num_edges * dims_x.num_feats \
            + dims_x.num_nodes * np.product(self.f_lin_neigh.weight.shape)
        return (
            Dims_X(dims_x.num_nodes, self.f_lin_self.weight.shape[0]),
            Dims_adj(dims_adj.num_nodes, dims_adj.num_edges)
        ), ops


class GIN(shaDowLayer):
    def __init__(self, dim_in, dim_out, dropout=0.0, act="relu", norm='norm_feat', eps=0, **kwargs):
        kwargs['norm_dim'] = (1, dim_out)
        super().__init__(dim_in, dim_out, dropout=dropout, act=act, norm=norm, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out, bias=True), 
            nn.ReLU(),
            nn.Linear(dim_out, dim_out, bias=True)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))
    
    def forward(self, inputs, sizes_subg):
        feat_in, adj, is_normed, dropedge = inputs
        assert not is_normed
        feat_in = self.f_dropout(feat_in)
        if type(adj) == sp.csr_matrix:
            adj = coo_scipy2torch(adj.tocoo()).to(feat_in.device)
            deg_orig = get_deg_torch_sparse(adj)
            # dropedge
            masked_indices = torch.floor(
                torch.rand(int(adj._values().size()[0] * dropedge)) * adj._values().size()[0]
            ).long()
            adj._values()[masked_indices] = 0
            deg_dropped = torch.clamp(get_deg_torch_sparse(adj), min=1)
            rescale = torch.repeat_interleave(deg_orig / deg_dropped, deg_orig.long())
            adj._values()[:] = adj._values() * rescale
        feat_aggr = torch.sparse.mm(adj, feat_in)
        feat_aggr += (1 + self.eps) * feat_in
        feat_out = self.mlp(feat_aggr)
        feat_out = self.f_norm([self.act(feat_out)], sizes_subg)[0]
        return feat_out, adj, False, 0.

    def complexity(self, dims_x, dims_adj):
        assert dims_x.num_nodes == dims_adj.num_nodes
        ops = dims_adj.num_edges * dims_x.num_feats
        # TODO
        ops += dims_x.num_nodes * dims_x.num_feats      # (1 + eps) * X
        for m in self.mlp:
            breakpoint()       # TODO compute complexity in MLP
        return 


class GAT(shaDowLayer):
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        dropout: float=0.0, 
        act: str="relu", 
        norm: str='norm_feat', 
        mulhead: int=1, 
        **kwargs
    ):
        self.mulhead = mulhead
        assert dim_out % self.mulhead == 0, "invalid output dimension: need to be divisible by mulhead"
        self.dim_slice = int(dim_out / self.mulhead)
        kwargs['norm_dim'] = (2, self.mulhead, self.dim_slice)
        super().__init__(dim_in, dim_out, dropout=dropout, act=act, norm=norm, **kwargs)
        self.att_act = nn.LeakyReLU(negative_slope=0.2)     # See original GAT paper
        self.f_lin = nn.ModuleList(nn.Linear(dim_in, dim_out, bias=True) for i in range(2))        # neigh + self
        self.attention = nn.Parameter(torch.ones(2, self.mulhead, self.dim_slice))
        nn.init.xavier_uniform_(self.attention)        

    def _aggregate_attention(self, adj, feat_neigh, feat_self, attention_self, attention_neigh):
        """
        Here we manually implement the numerically stable version of softmax, 
        supporting variable length neighbors. 
        
        The torch version of softmax only support fixed length neighbors, and
        thus cannot be directly used here. 
        """
        attention_self = self.att_act(attention_self.mm(feat_self.t())).squeeze()       # num_nodes
        attention_neigh = self.att_act(attention_neigh.mm(feat_neigh.t())).squeeze()    # num_nodes
        val_adj = (attention_self[adj._indices()[0]] + attention_neigh[adj._indices()[1]]) # * adj._values()
        # Compute softmax per neighborhood: substract max for stability in softmax computation
        max_per_row = scatter(val_adj, adj._indices()[0], reduce="max")     # here we may select some entry that will eventually be dropedged. But this is ok. 
        deg = scatter(torch.ones(val_adj.size()).to(feat_neigh.device), adj._indices()[0], reduce="sum")
        val_adj_norm = val_adj - torch.repeat_interleave(max_per_row, deg.long())
        val_adj_exp = torch.exp(val_adj_norm) * adj._values()
        # put coefficient alpha into the adj values
        att_adj = torch.sparse.FloatTensor(adj._indices(), val_adj_exp, torch.Size(adj.shape))
        denom = torch.clamp(scatter(val_adj_exp, adj._indices()[0], reduce="sum"), min=1e-10)
        # aggregate
        ret = self.spmm(att_adj, feat_neigh)
        ret *= 1 / denom.view(-1, 1)
        return ret

    def _adj_norm(self, adj, is_normed, device, dropedge=0):
        """
        Will perform edge dropout only when is_normed == False
        """
        if type(adj) == sp.csr_matrix:
            assert not is_normed
            adj_norm = coo_scipy2torch(adj.tocoo()).to(device)
            # here we don't normalize adj (data = 1,1,1...). In DGL, it is sym normed
            if dropedge > 0:
                masked_indices = torch.floor(
                    torch.rand(int(adj_norm._values().size()[0] * dropedge)) * adj_norm._values().size()[0]
                ).long()
                adj_norm._values()[masked_indices] = 0
        else:
            assert type(adj) == torch.Tensor and is_normed
            adj_norm = adj
        return adj_norm

    def forward(self, inputs, sizes_subg):
        feat_in, adj, is_normed, dropedge = inputs
        adj_norm = self._adj_norm(adj, is_normed, feat_in.device, dropedge=dropedge)
        feat_in = self.f_dropout(feat_in)
        # generate A^i X
        N = feat_in.shape[0]
        feat_partial_self  = self.act(self.f_lin[0](feat_in)).view(N, self.mulhead, -1)
        feat_partial_neigh = self.act(self.f_lin[1](feat_in)).view(N, self.mulhead, -1)
        feat_partial_self  = [feat_partial_self[:, t] for t in range(self.mulhead)]
        feat_partial_neigh = [feat_partial_neigh[:, t] for t in range(self.mulhead)]
        for k in range(self.mulhead):
            feat_partial_neigh[k] = self._aggregate_attention(
                adj_norm,
                feat_partial_neigh[k],
                feat_partial_self[k],
                self.attention[0, k].unsqueeze(0),
                self.attention[1, k].unsqueeze(0)
            )
        feat_partial_neigh, feat_partial_self = self.f_norm(
            [feat_partial_neigh, feat_partial_self], sizes_subg
        )
        feat_partial_self = torch.cat(tuple(feat_partial_self), dim=1)
        feat_partial_neigh = torch.cat(tuple(feat_partial_neigh), dim=1)
        feat_out = (feat_partial_self + feat_partial_neigh) / 2
        return feat_out, adj_norm, True, 0.

    def complexity(self, dims_X, dims_adj):
        assert dims_X.num_nodes == dims_adj.num_nodes
        ops = 0
        # ops for X W
        ops += dims_X.num_nodes * np.product(self.f_lin[0].weight.shape)
        ops += dims_X.num_nodes * np.product(self.f_lin[1].weight.shape)
        # ops for atten vector times (X W)
        ops += dims_X.num_nodes * self.f_lin[0].weight.shape[0]
        ops += dims_X.num_nodes * self.f_lin[1].weight.shape[0]
        for _h in range(self.mulhead):
            ops += dims_adj.num_edges * 2
            # ops for softmax (assume calculating softmax is of cost 20 -- 
            # exp and division are much more expensive)
            ops += dims_adj.num_edges * 20
        # ops for weighted aggregation
        ops += dims_adj.num_edges * self.f_lin[1].weight.shape[0]
        return (Dims_X(dims_X.num_nodes, self.f_lin[1].weight.shape[0]),
                Dims_adj(dims_adj.num_nodes, dims_adj.num_edges)), ops


class GATScatter(shaDowLayer):
    """
    The modified GAT based on DGL. Due to limited time, we haven't go through a careful
    hyperparameter tuning process for this architecture. But the functionality of this layer
    should be correct. 
    """
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        dropout: float=0.0, 
        act: str="relu", 
        norm: str='norm_feat', 
        mulhead: int=1, 
        **kwargs
    ):
        kwargs['norm_dim'] = (1, dim_out)
        super().__init__(dim_in, dim_out, dropout=dropout, act=act, norm=norm, **kwargs)
        self.mulhead = mulhead
        self.att_act = nn.LeakyReLU(negative_slope=0.2)     # See original GAT paper
        assert dim_out % self.mulhead == 0, "invalid output dimension: need to be divisible by mulhead"
        self.dim_slice = int(dim_out / self.mulhead)        # bias = False??
        self.f_lin = nn.ModuleList(nn.Linear(dim_in, dim_out, bias=True) for i in range(2))        # neigh + self
        self.attention = nn.Parameter(torch.FloatTensor(size=(1, self.mulhead, self.dim_slice)))
        self.reset_parameters(act)
    
    def reset_parameters(self, act):
        gain = nn.init.calculate_gain(act)
        nn.init.xavier_normal_(self.f_lin[0].weight, gain=gain)
        nn.init.xavier_normal_(self.f_lin[1].weight, gain=gain)
        nn.init.xavier_normal_(self.attention, gain=gain)

    def _adj_drop(self, adj, is_normed, device, dropedge=0):
        """
        Will perform edge dropout only when is_normed == False
        """
        if type(adj) == sp.csr_matrix:
            assert not is_normed
            adj_norm = coo_scipy2torch(adj.tocoo()).to(device)
            # here we don't normalize adj (data = 1,1,1...). In DGL, it is sym normed
            if dropedge > 0:
                masked_indices = torch.floor(torch.rand(int(adj_norm._values().size()[0] * dropedge)) * adj_norm._values().size()[0]).long()
                adj_norm._values()[masked_indices] = 0
        else:
            assert type(adj) == torch.Tensor and is_normed
            adj_norm = adj
        return adj_norm

    def _aggregate_attention(self, adj, feat_neigh, el_edges):
        val_adj = el_edges # * adj._values()
        # Compute softmax per neighborhood: substract max for stability in softmax computation
        max_per_row = scatter(val_adj, adj._indices()[0], reduce="max")
        deg = scatter(torch.ones(val_adj.size()).to(feat_neigh.device), adj._indices()[0], reduce="sum")
        val_adj_norm = val_adj - torch.repeat_interleave(max_per_row, deg.long())
        val_adj_exp = torch.exp(val_adj_norm) * adj._values()
        # put coefficient alpha into the adj values
        att_adj = torch.sparse.FloatTensor(adj._indices(), val_adj_exp, torch.Size(adj.shape))
        denom = torch.clamp(scatter(val_adj_exp, adj._indices()[0], reduce="sum"), min=1e-10)
        # aggregate
        ret = self.spmm(att_adj, feat_neigh)
        ret *= 1 / denom.view(-1, 1)
        return ret

    def forward(self, inputs, sizes_subg):
        feat_in, adj, is_dropped, dropedge = inputs

        # generate A^i X
        N = feat_in.shape[0]
        h_dst = h_src = self.f_dropout(feat_in)
        feat_src = self.f_lin[0](h_src).view(N, self.mulhead, -1)
        adj_drop = self._adj_drop(adj, is_dropped, feat_in.device, dropedge=dropedge)

        # sym-norm: part 1
        # degs = torch.bincount(adj_drop._indices()[0])           # ignore the dropped edges: since softmax will normalize anyways
        # norm = torch.pow(degs, -0.5)
        # shp = norm.shape + (1,) * (feat_src.dim() - 1)
        # norm = torch.reshape(norm, shp)
        # feat_src = feat_src * norm                              # num_nodes x mulhead x dim_slice

        el = (feat_src * self.attention).sum(dim=-1).unsqueeze(-1)
        el = self.att_act(el)
        el_edges = el[adj_drop._indices()[1]].squeeze(-1)       # num_edges x mulhead x 1

        feat_aggr = []
        for k in range(self.mulhead):
            feat_aggr.append(self._aggregate_attention(adj_drop, feat_src[:, k], el_edges[:, k]))
        feat_aggr = torch.cat(feat_aggr, dim=1)

        # sym-norm: part 2
        # feat_aggr = feat_aggr * norm.squeeze(-1)

        # residue
        feat_self = self.f_lin[1](h_dst)
        feat_out = self.f_norm([self.act(feat_aggr + feat_self)], sizes_subg)[0]
        # TODO: there seems to be one more dropout here?
        return feat_out, adj_drop, True, 0.
        
