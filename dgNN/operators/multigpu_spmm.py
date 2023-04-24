from torch.utils.cpp_extension import load
import torch
import spmm_multigpu

def MultigpuSpMM(
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
    return MultigpuSpMMFunction.apply(
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
    )


class MultigpuSpMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        out_feat = spmm_multigpu.multigpu_spmm(
            nnz, row_ptr_s, col_idx_s, edge_val_csr_s, p_csr, q_csr, in_feat
        )
        ctx.nnz = nnz
        ctx.col_ptr_s = col_ptr_s
        ctx.row_idx_s = row_idx_s
        ctx.edge_val_csc_s = edge_val_csc_s
        ctx.p_csc = p_csc
        ctx.q_csc = q_csc
        return out_feat

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        # print("start backward")
        grad_feat = spmm_multigpu.multigpu_spmm(
            ctx.nnz, ctx.col_ptr_s, ctx.row_idx_s, ctx.edge_val_csc_s, ctx.p_csc, ctx.q_csc, grad_out
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_feat,
        )
