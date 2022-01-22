import numpy as np
import torch
import torch.nn.functional as F
from dgl import function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from typing import List
from graph_engine.frontend import TRAIN, VALID, TEST


n_node_feats, n_classes = 0, 0


def load_data(dataset):
    global n_node_feats, n_classes

    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


def general_outcome_correlation(graph, y0, n_prop=50, alpha=0.8, use_norm=False, post_step=None):
    with graph.local_scope():
        y = y0
        for _ in range(n_prop):
            if use_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (y.dim() - 1)
                norm = torch.reshape(norm, shp)
                y = y * norm

            graph.srcdata.update({"y": y})
            graph.update_all(fn.copy_u("y", "m"), fn.mean("m", "y"))
            y = graph.dstdata["y"]

            if use_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (y.dim() - 1)
                norm = torch.reshape(norm, shp)
                y = y * norm

            y = alpha * y + (1 - alpha) * y0

            if post_step is not None:
                y = post_step(y)

        return y


def evaluate(labels, pred, train_idx, val_idx, test_idx, evaluator):
    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
    )


def run(use_norm, alpha, graph, labels, pred, train_idx, val_idx, test_idx, evaluator):
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]

    y = pred.clone()
    y[train_idx] = F.one_hot(labels[train_idx], n_classes).float().squeeze(1)
    # dy = torch.zeros(graph.number_of_nodes(), n_classes, device=device)
    # dy[train_idx] = F.one_hot(labels[train_idx], n_classes).float().squeeze(1) - pred[train_idx]

    train_acc, val_acc, test_acc = evaluate(labels, y, train_idx, val_idx, test_idx, evaluator_wrapper)
    acc_orig = {TRAIN: train_acc, VALID: val_acc, TEST: test_acc}
    # print("train acc:", _train_acc)
    print("original val acc:", val_acc)
    print("original test acc:", test_acc)

    # NOTE: Only "smooth" is performed here.
    # smoothed_dy = general_outcome_correlation(
    #     graph, dy, alpha=args.alpha1, use_norm=args.use_norm, post_step=lambda x: x.clamp(-1, 1)
    # )

    # y[train_idx] = F.one_hot(labels[train_idx], n_classes).float().squeeze(1)
    # smoothed_dy = smoothed_dy
    # y = y + args.alpha2 * smoothed_dy  # .clamp(0, 1)

    smoothed_y = general_outcome_correlation(
        graph, y, alpha=alpha, use_norm=use_norm, post_step=lambda x: x.clamp(0, 1)
    )

    train_acc, val_acc, test_acc = evaluate(labels, smoothed_y, train_idx, val_idx, test_idx, evaluator_wrapper)
    acc_cs = {TRAIN: train_acc, VALID: val_acc, TEST: test_acc}

    # print("train acc:", _train_acc)
    print("val acc:", val_acc)
    print("test acc:", test_acc)

    return acc_orig, acc_cs


def correct_smooth(dataset, dev_torch, pred_l : list, use_norm, alpha):

    if dataset == 'arxiv':
        dataset = 'ogbn-arxiv'
    elif dataset == 'products':
        dataset = 'ogbn-products'
    else:
        raise NotImplementedError
    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph = preprocess(graph)

    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(dev_torch), (graph, labels, train_idx, val_idx, test_idx)
    )

    acc_orig_l, acc_cs_l = [], []
    for pred in pred_l:
        acc_orig, acc_cs = run(use_norm, alpha, graph, labels, pred, train_idx, val_idx, test_idx, evaluator)
        acc_orig_l.append(acc_orig)
        acc_cs_l.append(acc_cs)
    return _merge_stat(acc_orig_l), _merge_stat(acc_cs_l)

def _merge_stat(dict_l : List[dict]):
    key_l = [set(d.keys()) for d in dict_l]
    assert all(k == key_l[0] for k in key_l)
    ret = {k: [] for k in key_l[0]}
    for d in dict_l:
        for k, v in d.items():
            ret[k].append(v)
    return ret
