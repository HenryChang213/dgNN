import torch
import multigpu_gcnconv
import time
import sys
import argparse
import copy
import random
import math

import dgl
from dgl.data.reddit import RedditDataset
import scipy.sparse as sp
from ogb.nodeproppred import DglNodePropPredDataset


def load_dataset(args):
    if args.dataset == "cora":
        data = dgl.data.CoraGraphDataset()
        g = data[0]
    elif args.dataset == "reddit":
        data = dgl.data.RedditDataset()
        g = data[0]
        baseline_time = 1
    elif args.dataset == "ogbn-arxiv":
        data = DglNodePropPredDataset(name="ogbn-arxiv", root="/home/hz0567/.ogb")
        g, _ = data[0]
        baseline_time = 1
    elif args.dataset == "ogbn-products":
        data = DglNodePropPredDataset(name="ogbn-products", root="/home/hz0567/.ogb")
        g, _ = data[0]
        baseline_time = 1
    elif args.dataset == "ogbn-papers100M":
        data = DglNodePropPredDataset(name="ogbn-papers100M", root="/home/hz0567/.ogb")
        g, _ = data[0]
        baseline_time = 1
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

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

    adj_csr = sp.csr_matrix(
        (torch.ones(row.shape), (row, col)), shape=(g.num_nodes(), g.num_nodes())
    )

    row_ptr = torch.from_numpy(adj_csr.indptr)
    col_idx = torch.from_numpy(adj_csr.indices)

    adj_csc = adj_csr.tocsc()

    col_ptr = torch.from_numpy(adj_csc.indptr)
    row_idx = torch.from_numpy(adj_csc.indices)

    # prepare weight
    weight_s = [torch.randn(f, num_hidden, dtype=torch.float32)]
    for i in range(1, num_layers - 1):
        weight_s.append(torch.randn(num_hidden, num_hidden, dtype=torch.float32))
    weight_s.append(torch.randn(num_hidden, num_classes, dtype=torch.float32))

    return (
        m,
        nnz,
        col,
        row,
        edge_val,
        in_feat,
        weight_s,
        num_classes,
        row_ptr,
        col_idx,
        col_ptr,
        row_idx,
    )


def partion_graph(args, row, col, edge_val, m, p, q):
    num_gpu = torch.cuda.device_count()

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
    (
        m,
        nnz,
        col,
        row,
        edge_val,
        in_feat,
        weight_s,
        num_classes,
        row_ptr,
        col_idx,
        col_ptr,
        row_idx,
    ) = load_dataset(args)

    numiter = 100

    if args.partition_alg == "sa":
        initial_temp = 90
        final_temp = 1
        alpha = 1
        # maxiter = 1000

        current_temp = initial_temp

        # p, q initial
        # p = torch.tensor([0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64)
        # q = torch.tensor([0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64)

        p = torch.searchsorted(
            row_ptr,
            torch.tensor(
                [0, int(nnz / 4), int(nnz / 2), int(nnz * 3 / 4), nnz],
                dtype=torch.int64,
            ),
        )
        q = torch.searchsorted(
            col_ptr,
            torch.tensor(
                [0, int(nnz / 4), int(nnz / 2), int(nnz * 3 / 4), nnz],
                dtype=torch.int64,
            ),
        )

        initial_state = [p, q]

        # Start by initializing the current state with the initial state
        current_state = initial_state
        solution = current_state

        row_ptr_s, col_idx_s, edge_val_s, _, _ = partion_graph(
            args, row, col, edge_val, m, initial_state[0], initial_state[1]
        )
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
            args.memory_manage,
        )
        torch.cuda.synchronize()

        current_cost = torch.mean(inference_time).item()
        solution_cost = current_cost

        while current_temp > final_temp:
            perturb = copy.deepcopy(current_state)
            perturb[0][1:4] += (torch.randn(3) * 100000).to(torch.int64)
            perturb[1][1:4] += (torch.randn(3) * 100000).to(torch.int64)

            # Constraining inputs
            for i in range(1, 4):
                if perturb[0][i] < perturb[0][i - 1]:
                    perturb[0][i] = perturb[0][i - 1]
                if perturb[1][i] < perturb[1][i - 1]:
                    perturb[1][i] = perturb[1][i - 1]
            for i in range(3, 1, -1):
                if perturb[0][i] > perturb[0][i + 1]:
                    perturb[0][i] = perturb[0][i + 1]
                if perturb[1][i] > perturb[1][i + 1]:
                    perturb[1][i] = perturb[1][i + 1]

            row_ptr_s, col_idx_s, edge_val_s, _, _ = partion_graph(
                args, row, col, edge_val, m, initial_state[0], initial_state[1]
            )
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
                args.memory_manage,
            )
            torch.cuda.synchronize()

            new_cost = torch.mean(inference_time).item()

            cost_diff = new_cost - current_cost

            if cost_diff < 0 or random.random() < math.exp(
                -cost_diff * 10 / current_temp
            ):
                current_state = perturb
                current_cost = new_cost
            if current_cost < solution_cost:
                solution = current_state
                solution_cost = current_cost

            print(
                "Temp {} deg: {} (best: {})".format(
                    current_temp, current_cost, solution_cost
                ),
                flush=True,
            )

            current_temp -= alpha

        print(solution)
        print(solution_cost)

    elif args.partition_alg in ["equal", "binary_search"]:
        if args.partition_alg == "equal":
            p = torch.tensor(
                [0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64
            )
            q = torch.tensor(
                [0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64
            )

        elif args.partition_alg == "binary_search":
            nnz = row_ptr[-1]

            p = torch.searchsorted(
                row_ptr,
                torch.tensor(
                    [0, int(nnz / 4), int(nnz / 2), int(nnz * 3 / 4), nnz],
                    dtype=torch.int64,
                ),
            )
            q = torch.searchsorted(
                col_ptr,
                torch.tensor(
                    [0, int(nnz / 4), int(nnz / 2), int(nnz * 3 / 4), nnz],
                    dtype=torch.int64,
                ),
            )

        print(p)
        print(q)

        for memory_manage in [0, 1, 2]:
            row_ptr_s, col_idx_s, edge_val_s, p, q = partion_graph(
                args, row, col, edge_val, m, p, q
            )
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
                memory_manage,
            )
            torch.cuda.synchronize()
            print(inference_time)

            speedup = torch.mean(inference_time).item()/baseline_time
            with open("{}".format(args.output), "a") as f:
                f.write(
                    "{},{},{},{},{},{},{}\n".format(
                        args.num_layers,
                        args.num_hidden,
                        torch.cuda.device_count(),
                        args.dataset,
                        memory_manage,
                        args.partition_alg,
                        speedup,
                    )
                )

    else:
        print("Unrecognized Partition Method: {}. Abort.".format(args.partition_alg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_hidden", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="reddit")
    parser.add_argument("--memory-manage", type=int, default=0)
    parser.add_argument("--partition-alg", type=str, default="sa")
    parser.add_argument("--output", type=str, default="output.txt")
    args = parser.parse_args()
    print(args)
    main(args)
