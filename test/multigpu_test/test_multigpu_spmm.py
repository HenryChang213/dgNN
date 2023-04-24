import torch
import spmm_multigpu
import time


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
f = 32
in_feat = torch.randn(m, f, dtype=torch.float32)
adj_coo = sp.coo_matrix((edge_val, (row, col)), shape=(m, m))
out_feat_gt = adj_coo.dot(in_feat)
# out_feat_gt = dgl.ops.u_mul_e_sum(dgl.from_scipy(adj_coo).to(0), in_feat.to(0), edge_val.to(0))
# g=g.to(0)
# with g.local_scope():
#     g.edata['a']=edge_val.to(0)
#     g.srcdata.update({'ft':in_feat.to(0)})
#     g.update_all(dgl.function.u_mul_e('ft','a','m'),dgl.function.sum('m','ft'))
#     out_feat_gt=g.dstdata['ft']

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

start=time.time()
out_feat_our = spmm_multigpu.multigpu_spmm(
    nnz, row_ptr_s, col_idx_s, edge_val_s, p, q, in_feat
)
torch.cuda.synchronize()
end=time.time()
print("spmm time",end-start)

print(out_feat_our[0])
print(out_feat_gt[0])
print(torch.max(torch.abs(out_feat_our - out_feat_gt)))
print(torch.min(torch.abs(out_feat_our - out_feat_gt)))
print(torch.mean(torch.abs(out_feat_our - out_feat_gt)))
