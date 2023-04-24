from torch import nn
import torch
from torch.nn.modules.linear import Identity

from ..operators.multigpu_spmm import MultigpuSpMM


class MultigpuGCNConv(nn.Module):  # our gat layer
    def __init__(self, in_feats, out_feats, activation=torch.nn.functional.relu):
        super(MultigpuGCNConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.FloatTensor(size=(out_feats,)))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.W is not None:
            torch.nn.init.xavier_uniform_(self.W)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

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
        in_feat,
    ):
        h = torch.matmul(in_feat, self.W)
        rst = MultigpuSpMM(
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
        rst = rst + self.bias
        if self.activation:
            rst = self.activation(rst)

        return rst
