import torch
import multigpu_gcnconv
import time
import sys
import argparse

import dgl
from dgl.data.reddit import RedditDataset
import scipy.sparse as sp


def load_dataset(args):
    if args.dataset == "cora":
        data = dgl.data.CoraGraphDataset()
    elif args.dataset == "reddit":
        data = dgl.data.RedditDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    m = g.num_nodes()
    nnz = g.num_edges()
    col, row = g.edges(order="srcdst")
    edge_val = torch.randn(col.shape, dtype=torch.float32)
    in_feat = g.ndata["feat"]
    f = in_feat.shape[1]

    num_layers = args.num_layers
    num_hidden = args.num_hidden
    num_classes = data.num_classes

    # prepare weight
    weight_s = [torch.randn(f, num_hidden, dtype=torch.float32)]
    for i in range(1, num_layers - 1):
        weight_s.append(torch.randn(num_hidden, num_hidden, dtype=torch.float32))
    weight_s.append(torch.randn(num_hidden, num_classes, dtype=torch.float32))

    return m, nnz, col, row, edge_val, in_feat, weight_s, num_classes


def partion_graph(args, row, col, edge_val, m):
    num_gpu = torch.cuda.device_count()

    # TODO
    p = torch.tensor([0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64)
    q = torch.tensor([0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64)

    row_ptr_s = []
    col_idx_s = []
    edge_val_s = []
    for i in range(num_gpu):
        for j in range(num_gpu):
            row_start = p[i]
            row_end = p[i + 1]
            col_start = q[j]
            col_end = q[j + 1]
            mask = (
                (row >= row_start)
                * (row < row_end)
                * (col >= col_start)
                * (col < col_end)
            )
            adj_csr = sp.csr_matrix(
                (edge_val[mask], (row[mask] - row_start, col[mask] - col_start)),
                shape=(row_end - row_start, col_end - col_start),
            )
            row_ptr_split = torch.from_numpy(adj_csr.indptr)
            col_idx_split = torch.from_numpy(adj_csr.indices)
            edge_val_split = torch.from_numpy(adj_csr.data)
            row_ptr_s.append(row_ptr_split)
            col_idx_s.append(col_idx_split)
            edge_val_s.append(edge_val_split)

    return row_ptr_s, col_idx_s, edge_val_s, p, q


def main(args):
    m, nnz, col, row, edge_val, in_feat, weight_s, num_classes = load_dataset(args)
    row_ptr_s, col_idx_s, edge_val_s, p, q = partion_graph(args, row, col, edge_val, m)
    out_feat_our, inference_time = multigpu_gcnconv.multigpu_gcn_inference(
        nnz,
        args.num_layers,
        args.num_hidden,
        num_classes,
        row_ptr_s,
        col_idx_s,
        edge_val_s,
        p,
        q,
        in_feat,
        weight_s,
        1,
    )
    torch.cuda.synchronize()
    print(inference_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_hidden", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cora")

    args = parser.parse_args()
    print(args)
    main(args)
