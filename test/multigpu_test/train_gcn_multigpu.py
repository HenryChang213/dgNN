# from dgl.nn.pytorch.conv import GATConv
from dgNN.layers.gcnconv_layer import MultigpuGCNConv
import torch.nn.functional as F
import torch
import time
import argparse
import dgl
import GPUtil
import scipy.sparse as sp
import sys

class Net(torch.nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, num_classes, activation=None):
        super().__init__()
        self.num_layers = num_layers
        self.gcn_layers = torch.nn.ModuleList()
        # input projection (no residual)
        self.gcn_layers.append(MultigpuGCNConv(in_dim, num_hidden, activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gcn_layers.append(MultigpuGCNConv(num_hidden, num_hidden, activation))
        # output projection
        self.gcn_layers.append(MultigpuGCNConv(num_hidden, num_classes, activation))

    def forward(
        self,
        nnz,
        row_ptr_s,
        col_idx_s,
        edge_val_csr_s,
        p_csr,
        q_csr,
        col_ptr_s,
        row_idx_s,
        edge_val_csc_s,
        p_csc,
        q_csc,
        x,
    ):
        h = x
        for l in range(self.num_layers):
            h = self.gcn_layers[l](
                nnz,
                row_ptr_s,
                col_idx_s,
                edge_val_csr_s,
                p_csr,
                q_csr,
                col_ptr_s,
                row_idx_s,
                edge_val_csc_s,
                p_csc,
                q_csc,
                h,
            )
        # output projection
        logits = self.gcn_layers[-1](
            nnz,
            row_ptr_s,
            col_idx_s,
            edge_val_csr_s,
            p_csr,
            q_csr,
            col_ptr_s,
            row_idx_s,
            edge_val_csc_s,
            p_csc,
            q_csc,
            h,
        )
        return logits


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def load_dataset(args):
    if args.dataset == "cora":
        data = dgl.data.CoraGraphDataset()
    elif args.dataset == "citeseer":
        data = dgl.data.CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        data = dgl.data.PubmedGraphDataset()
    elif args.dataset == "reddit":
        data = dgl.data.RedditDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    m = g.num_nodes()
    nnz = g.num_edges()
    col, row = g.edges(order="srcdst")
    edge_val = torch.ones(col.shape, dtype=torch.float32)
    p_csr = torch.tensor(
        [0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64
    )
    q_csr = torch.tensor(
        [0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64
    )
    adj_coo = sp.coo_matrix((edge_val, (row, col)), shape=(m, m))

    row_ptr_s = []
    col_idx_s = []
    edge_val_csr_s = []
    for i in range(args.n_gpus):
        for j in range(args.n_gpus):
            row_start = p_csr[i]
            row_end = p_csr[i + 1]
            col_start = q_csr[j]
            col_end = q_csr[j + 1]
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
            edge_val_csr_s.append(edge_val_split)

    p_csc = torch.tensor(
        [0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64
    )
    q_csc = torch.tensor(
        [0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64
    )
    adj_coo_T = adj_coo.transpose()
    row = torch.from_numpy(adj_coo_T.row)
    col = torch.from_numpy(adj_coo_T.col)
    edge_val = torch.from_numpy(adj_coo_T.data)
    col_ptr_s = []
    row_idx_s = []
    edge_val_csc_s = []
    for i in range(args.n_gpus):
        for j in range(args.n_gpus):
            row_start = p_csc[i]
            row_end = p_csc[i + 1]
            col_start = q_csc[j]
            col_end = q_csc[j + 1]
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
            col_ptr_s.append(row_ptr_split)
            row_idx_s.append(col_idx_split)
            edge_val_csc_s.append(edge_val_split)

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    n_feats = features.shape[1]
    n_classes = data.num_classes
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]

    return (
        nnz,
        row_ptr_s,
        col_idx_s,
        edge_val_csr_s,
        p_csr,
        q_csr,
        col_ptr_s,
        row_idx_s,
        edge_val_csc_s,
        p_csc,
        q_csc,
        features,
        labels,
        n_feats,
        n_classes,
        train_mask,
        test_mask,
    )


def main(args):
    # load dataset
    (
        nnz,
        row_ptr_s,
        col_idx_s,
        edge_val_csr_s,
        p_csr,
        q_csr,
        col_ptr_s,
        row_idx_s,
        edge_val_csc_s,
        p_csc,
        q_csc,
        features,
        labels,
        n_feats,
        n_classes,
        train_mask,
        test_mask,
    ) = load_dataset(args)
    
    print("finish data loading")
    sys.stdout.flush()

    model = Net(
        args.n_layers, n_feats, args.n_hidden, n_classes, torch.nn.functional.relu
    )
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # print("warm up")
    # maxMemory = 0
    # for _ in range(10):
    #     model.train()
    #     logits = model(
    #         nnz,
    #         row_ptr_s,
    #         col_idx_s,
    #         edge_val_csr_s,
    #         p_csr,
    #         q_csr,
    #         col_ptr_s,
    #         row_idx_s,
    #         edge_val_csc_s,
    #         p_csc,
    #         q_csc,
    #         features,
    #     )
    #     loss = loss_fcn(logits[train_mask], labels[train_mask])
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    print("profile training")
    model.train()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.n_epochs):
        logits = model(
            nnz,
            row_ptr_s,
            col_idx_s,
            edge_val_csr_s,
            p_csr,
            q_csr,
            col_ptr_s,
            row_idx_s,
            edge_val_csc_s,
            p_csc,
            q_csc,
            features,
        )
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        sys.stdout.flush()
    torch.cuda.synchronize()
    end = time.time()
    train_time = (end - start) / args.n_epochs

    print("profile inference")
    model.eval()
    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.n_epochs):
        with torch.no_grad():
            logits = model(
                nnz,
                row_ptr_s,
                col_idx_s,
                edge_val_csr_s,
                p_csr,
                q_csr,
                col_ptr_s,
                row_idx_s,
                edge_val_csc_s,
                p_csc,
                q_csc,
                features,
            )
    torch.cuda.synchronize()
    end = time.time()
    inference_time = (end - start) / args.n_epochs

    logits = logits[test_mask]
    acc = accuracy(logits, labels[test_mask])
    print("Test Accuracy {:.4f}".format(acc))
    print("train time:", train_time)
    print("inference time:", inference_time)

    # if args.output != None:
    #     with open("{}".format(args.output), "a") as f:
    #         print(
    #             "train_GAT_dgl,{} heads={} hidden_dim={},{:f}s,{:f}s,{}MB,{}".format(
    #                 args.dataset,
    #                 args.n_heads,
    #                 args.n_hidden,
    #                 train_time,
    #                 inference_time,
    #                 maxMemory,
    #                 acc,
    #             ),
    #             file=f,
    #         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--dataset", type=str, default="reddit")
    parser.add_argument("--n-gpus", type=int, default=4, help="number of GPUs to use")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=20, help="number of training epochs"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=128, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=5, help="number of hidden gcn layers"
    )
    parser.add_argument("--output", type=str, default=None, help="output file")

    args = parser.parse_args()
    print(args)
    main(args)
