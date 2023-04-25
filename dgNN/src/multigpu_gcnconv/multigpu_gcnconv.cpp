#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

torch::Tensor multigpu_spmm_cuda(
    const long nnz,
    std::vector<torch::Tensor> row_ptr_s,   // int
    std::vector<torch::Tensor> col_idx_s,   // int
    std::vector<torch::Tensor> edge_val_s,  // float
    torch::Tensor p, torch::Tensor q,       // long
    torch::Tensor in_feat                   // float
);

torch::Tensor multigpu_spmm(const long nnz,
                            std::vector<torch::Tensor> row_ptr_s,   // int
                            std::vector<torch::Tensor> col_idx_s,   // int
                            std::vector<torch::Tensor> edge_val_s,  // float
                            torch::Tensor p, torch::Tensor q,       // long
                            torch::Tensor in_feat                   // float
) {
  assert(row_ptr_s.size() == col_idx_s.size());
  assert(row_ptr_s.size() == edge_val_s.size());
  assert(row_ptr_s.size() == p.size(0) - 1);
  assert(p.size(0) == q.size(0));
  for (size_t i = 0; i < row_ptr_s.size(); i++) {
    assert(row_ptr_s[i].device().type() == torch::kCPU);
    assert(col_idx_s[i].device().type() == torch::kCPU);
    assert(edge_val_s[i].device().type() == torch::kCPU);
    assert(row_ptr_s[i].is_contiguous());
    assert(col_idx_s[i].is_contiguous());
    assert(edge_val_s[i].is_contiguous());
    assert(row_ptr_s[i].dtype() == torch::kInt32);
    assert(col_idx_s[i].dtype() == torch::kInt32);
    assert(edge_val_s[i].dtype() == torch::kFloat32);
  }
  assert(p.device().type() == torch::kCPU);
  assert(q.device().type() == torch::kCPU);
  assert(in_feat.device().type() == torch::kCPU);
  assert(p.is_contiguous());
  assert(q.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(p.dtype() == torch::kInt64);
  assert(q.dtype() == torch::kInt64);
  assert(in_feat.dtype() == torch::kFloat32);
  return multigpu_spmm_cuda(nnz, row_ptr_s, col_idx_s, edge_val_s, p, q,
                            in_feat);
}

torch::Tensor multigpu_gcn_inference_cuda(
    const long nnz, const int num_layers, const int num_hidden,
    const int num_classes,
    const std::vector<torch::Tensor> row_ptr_s,    // int
    const std::vector<torch::Tensor> col_idx_s,    // int
    const std::vector<torch::Tensor> edge_val_s,   // float
    const torch::Tensor p, const torch::Tensor q,  // long
    const torch::Tensor in_feat,                   // float
    const std::vector<torch::Tensor> weight_s      // float
);

torch::Tensor multigpu_gcn_inference(
    const long nnz, const int num_layers, const int num_hidden,
    const int num_classes,
    const std::vector<torch::Tensor> row_ptr_s,    // int
    const std::vector<torch::Tensor> col_idx_s,    // int
    const std::vector<torch::Tensor> edge_val_s,   // float
    const torch::Tensor p, const torch::Tensor q,  // long
    const torch::Tensor in_feat,                   // float
    const std::vector<torch::Tensor> weight_s      // float
) {
  assert(row_ptr_s.size() == col_idx_s.size());
  assert(row_ptr_s.size() == edge_val_s.size());
  assert(row_ptr_s.size() == p.size(0) - 1);
  assert(p.size(0) == q.size(0));
  for (size_t i = 0; i < row_ptr_s.size(); i++) {
    assert(row_ptr_s[i].device().type() == torch::kCPU);
    assert(col_idx_s[i].device().type() == torch::kCPU);
    assert(edge_val_s[i].device().type() == torch::kCPU);
    assert(row_ptr_s[i].is_contiguous());
    assert(col_idx_s[i].is_contiguous());
    assert(edge_val_s[i].is_contiguous());
    assert(row_ptr_s[i].dtype() == torch::kInt32);
    assert(col_idx_s[i].dtype() == torch::kInt32);
    assert(edge_val_s[i].dtype() == torch::kFloat32);
  }
  assert(p.device().type() == torch::kCPU);
  assert(q.device().type() == torch::kCPU);
  assert(in_feat.device().type() == torch::kCPU);
  assert(p.is_contiguous());
  assert(q.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(p.dtype() == torch::kInt64);
  assert(q.dtype() == torch::kInt64);
  assert(in_feat.dtype() == torch::kFloat32);
  for (size_t i = 0; i < weight_s.size(); i++) {
    assert(weight_s[i].device().type() == torch::kCPU);
    assert(weight_s[i].is_contiguous());
    assert(weight_s[i].dtype() == torch::kFloat32);
  }
  return multigpu_gcn_inference_cuda(nnz, num_layers, num_hidden, num_classes,
                                     row_ptr_s, col_idx_s, edge_val_s, p, q,
                                     in_feat, weight_s);
}
PYBIND11_MODULE(multigpu_gcnconv, m) {
  m.doc() = "multi-gpu gcnconv kernel. ";
  m.def("multigpu_spmm", &multigpu_spmm, "multi-gpu spmm op");
  m.def("multigpu_gcn_inference", &multigpu_gcn_inference,
        "multi-gpu gcn inference op");
}