import torch
import multigpu_gcnconv
import time
import sys

import dgl
from dgl.data.reddit import RedditDataset
import scipy.sparse as sp

data = RedditDataset()
g = data[0]
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
m = g.num_nodes()
nnz = g.num_edges()
col, row = g.edges(order="srcdst")
edge_val = torch.randn(col.shape, dtype=torch.float32)
p = torch.tensor([0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64)
q = torch.tensor([0, int(m / 4), int(m / 2), int(m * 3 / 4), m], dtype=torch.int64)
num_layers = 2
num_hidden = 16
num_classes = data.num_classes
in_feat = g.ndata["feat"]
f = in_feat.shape[1]
in_feat = torch.randn(m, f, dtype=torch.float32)
print(
    "num_layers",
    num_layers,
    "num_hidden",
    num_hidden,
    "num_classes",
    num_classes,
    "f",
    f,
)
sys.stdout.flush()

# prepare weight
weight_s = [torch.randn(f, num_hidden, dtype=torch.float32)]
for i in range(1, num_layers - 1):
    weight_s.append(torch.randn(num_hidden, num_hidden, dtype=torch.float32))
weight_s.append(torch.randn(num_hidden, num_classes, dtype=torch.float32))

# compute ground truth
adj_coo = sp.coo_matrix((edge_val, (row, col)), shape=(m, m))
# first layer
out_feat_gt = torch.from_numpy(adj_coo.dot(in_feat))
out_feat_gt = torch.matmul(out_feat_gt, weight_s[0])
out_feat_gt = torch.nn.functional.relu(out_feat_gt)
for i in range(1, num_layers - 1):
    out_feat_gt = torch.from_numpy(adj_coo.dot(out_feat_gt))
    out_feat_gt = torch.matmul(out_feat_gt, weight_s[i])
    out_feat_gt = torch.nn.functional.relu(out_feat_gt)
out_feat_gt = torch.from_numpy(adj_coo.dot(out_feat_gt))
out_feat_gt = torch.matmul(out_feat_gt, weight_s[-1])
out_feat_gt = torch.nn.functional.relu(out_feat_gt)
print("finish ground truth")
sys.stdout.flush()


row_ptr_s = []
col_idx_s = []
edge_val_s = []
for i in range(4):
    for j in range(4):
        row_start = p[i]
        row_end = p[i + 1]
        col_start = q[j]
        col_end = q[j + 1]
        mask = (
            (row >= row_start) * (row < row_end) * (col >= col_start) * (col < col_end)
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

start = time.time()
out_feat_our = multigpu_gcnconv.multigpu_gcn_inference(
    nnz,
    num_layers,
    num_hidden,
    num_classes,
    row_ptr_s,
    col_idx_s,
    edge_val_s,
    p,
    q,
    in_feat,
    weight_s,
)
torch.cuda.synchronize()
end = time.time()
print("gcnconv time", end - start)

print(out_feat_our[0])
print(out_feat_gt[0])
diff = out_feat_our - out_feat_gt
flat_index_of_biggest_element = diff.argmax()
x, y = (
    flat_index_of_biggest_element // diff.shape[1],
    flat_index_of_biggest_element % diff.shape[1],
)
print(x, y)
print(out_feat_our[x])
print(out_feat_gt[x])
max_diff = torch.max(torch.abs(out_feat_our - out_feat_gt))
print(max_diff)
print(torch.min(torch.abs(out_feat_our - out_feat_gt)))
print(torch.mean(torch.abs(out_feat_our - out_feat_gt)))
print(max_diff / torch.mean(out_feat_gt))
