import scipy.sparse as sp
import numpy as np
import torch
from torch_scatter import scatter


def get_adj_dtype(adj=None, num_nodes=-1, num_edges=-1):
    if adj is not None:
        num_nodes, num_edges = adj.shape[0], adj.size
    else:
        assert num_edges > 0 and num_nodes > 0
    return np.uint32 if max(num_nodes, num_edges) < 2**32 else np.int64


def to_undirected_csr(adj):
    """
    input adj is in csr format
    returned adj is in csr format
    """
    print("Converting graph to undirected. This may take a while for large graphs ...")
    if not (adj.data.max() == adj.data.min() == 1):
        adj.data[:] = 1
        print("[WARNING]: DISCARDING ALL EDGE VALUES WHEN TRANSFORMING TO UNDIRECTED GRAPH!!")
    adj_coo = adj.tocoo()
    adj_trans = sp.coo_matrix((adj_coo.data, (adj_coo.col, adj_coo.row)), shape=adj_coo.shape)
    adj_trans = adj_trans.tocsr()
    indptr_new = np.zeros(adj.indptr.size, dtype=np.int64)
    indices_new = []
    for i in range(adj.shape[0]):
        neigh1 = adj.indices[adj.indptr[i] : adj.indptr[i + 1]]
        neigh2 = adj_trans.indices[adj_trans.indptr[i] : adj_trans.indptr[i + 1]]
        neigh_merged = np.union1d(neigh1, neigh2)
        indptr_new[i + 1] = indptr_new[i] + neigh_merged.size
        indices_new.append(neigh_merged)
    indices_new = np.concatenate(indices_new)
    data_new = np.broadcast_to(np.ones(1, dtype=np.bool), indices_new.size)
    adj_und = sp.csr_matrix((data_new, indices_new, indptr_new), shape=adj.shape)
    dtype = get_adj_dtype(adj=adj_und)
    adj_und.indptr = adj_und.indptr.astype(dtype, copy=False)
    adj_und.indices = adj_und.indices.astype(dtype, copy=False)
    return adj_und


def coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values).type(torch.get_default_dtype())
    return torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))


# =============== #
#    ADJ UTILS    #
# =============== #

def get_deg_torch_sparse(adj):
    return scatter(adj._values(), adj._indices()[0], reduce="sum")


def adj_norm_rw(adj, deg=None, dropedge=0., sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    
    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    """
    if type(adj) == torch.Tensor:
        assert deg is None
        assert torch.sum(adj._values()).cpu().long().item() == adj._values().size()[0]
        _deg_orig = get_deg_torch_sparse(adj)
        if dropedge > 0:
            num_dropped_edges = int(adj._values().size()[0] * dropedge)
            masked_indices = torch.floor(torch.rand(num_dropped_edges) * adj._values().size()[0]).long()
            adj._values()[masked_indices] = 0
            _deg_dropped = get_deg_torch_sparse(adj)
        else:
            _deg_dropped = _deg_orig
        _deg = torch.repeat_interleave(_deg_dropped, _deg_orig.long())
        _val = adj._values()
        _val /= torch.clamp(_deg, min=1)
        adj_norm = adj
    else:
        assert dropedge == 0., "not supporting dropedge for scipy csr matrices"
        assert adj.shape[0] == adj.shape[1]
        diag_shape = (adj.shape[0], adj.shape[1])
        D = adj.sum(1).flatten() if deg is None else deg
        D = np.clip(D, 1, None)     # if deg_v == 0, it doesn't matter what value we clip it to. 
        norm_diag = sp.dia_matrix((1 / D, 0), shape=diag_shape)
        adj_norm = norm_diag.dot(adj)
        if sort_indices:
            adj_norm.sort_indices()
    return adj_norm


def adj_norm_sym(adj, sort_indices=True, add_self_edge=False, dropedge=0.):
    assert adj.shape[0] == adj.shape[1]
    assert adj.data.sum() == adj.size, "symmetric normalization only supports binary input adj"
    N = adj.shape[0]
    # drop edges symmetrically
    if dropedge > 0:
        masked_indices = np.random.choice(adj.size, int(adj.size * dropedge))
        data_m = adj.data.copy()
        data_m[masked_indices] = 0
        adj_m = sp.csr_matrix((data_m, adj.indices, adj.indptr), shape=adj.shape)
        data_add = adj_m.data + adj_m.tocsc().data
        survived_indices = np.where(data_add == 2)[0]
        data_m[:] = 0
        data_m[survived_indices] = 1
        adj = sp.csr_matrix((data_m, adj.indices, adj.indptr), shape=adj.shape)
    # augment adj with self-connection
    if add_self_edge:
        indptr_new = np.zeros(N + 1)
        neigh_list = [set(adj.indices[adj.indptr[v] : adj.indptr[v+1]]) for v in range(N)]
        for i in range(len(neigh_list)):
            neigh_list[i].add(i)
            neigh_list[i] = np.sort(np.fromiter(neigh_list[i], int, len(neigh_list[i])))
            indptr_new[i + 1] = neigh_list[i].size
        indptr_new = indptr_new.cumsum()
        indices_new = np.concatenate(neigh_list)
        data_new = np.broadcast_to(np.ones(1), indices_new.size)
        adj_aug = sp.csr_matrix((data_new, indices_new, indptr_new), shape=adj.shape)
        # NOTE: no need to explicitly convert dtype, since adj_norm_sym is used for subg only
    else:
        adj_aug = adj
    # normalize
    D = np.clip(adj_aug.sum(1).flatten(), 1, None)
    norm_diag = sp.dia_matrix((np.power(D, -0.5), 0), shape=adj_aug.shape)
    adj_norm = norm_diag.dot(adj_aug).dot(norm_diag)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm
