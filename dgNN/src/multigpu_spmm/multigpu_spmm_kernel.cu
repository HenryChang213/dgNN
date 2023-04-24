#include <cuda.h>
#include <nccl.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

#include "../util/computeUtil.h"

#define MULTIGPU_SPMM_DEBUG 0

//(m, 1, 1), (32, (f + 31) / 32, 1)
__global__ void spmm_kernel(const int f, const int* row_ptr, const int* col_idx,
                            const float* edge_val, const float* in_feat,
                            float* out_feat, const int stage) {
  int rid = blockIdx.x;
  int fid = threadIdx.y * blockDim.x + threadIdx.x;
  // extern __shared__ float edge_val_sh[];

  int lb = row_ptr[rid];
  int hb = row_ptr[rid + 1];
  int cid;
  float acc = 0;

  if (fid < f) {
    for (int i = lb; i < hb; i++) {
      cid = col_idx[i];
      acc += edge_val[i] * in_feat[cid * f + fid];
    }
    // printf("nnz=%d, before (%d,%d):%f\n", hb - lb, rid, fid,
    //  out_feat[rid * f + fid]);
    if (stage) {
      out_feat[rid * f + fid] += acc;
    } else {
      out_feat[rid * f + fid] = acc;
    }
    // printf("after (%d,%d):%f\n", rid, fid, out_feat[rid * f + fid]);
  }
}

torch::Tensor multigpu_spmm_cuda(
    const long nnz,
    std::vector<torch::Tensor> row_ptr_s,   // int
    std::vector<torch::Tensor> col_idx_s,   // int
    std::vector<torch::Tensor> edge_val_s,  // float
    torch::Tensor p, torch::Tensor q,       // long
    torch::Tensor in_feat                   // float
) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  // std::cout << "deviceCount:" << deviceCount << std::endl;
  // std::cout << "row_ptr_s.size:" << row_ptr_s.size() << std::endl;
  assert(deviceCount * deviceCount == row_ptr_s.size());
  assert(deviceCount * deviceCount == col_idx_s.size());
  assert(deviceCount * deviceCount == edge_val_s.size());
  assert(deviceCount + 1 == p.size(0));
  assert(deviceCount + 1 == q.size(0));
  const int m = in_feat.size(0);
  const int f = in_feat.size(1);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto out_feat = torch::empty({m, f}, options);

  if (MULTIGPU_SPMM_DEBUG) {
    for (int i = 0; i < deviceCount; i++) {
      for (int j = 0; j < deviceCount; j++) {
        std::cout << "(" << i << "," << j << ")\n";
        std::cout << row_ptr_s[i * deviceCount + j] << std::endl;
        std::cout << col_idx_s[i * deviceCount + j] << std::endl;
        std::cout << edge_val_s[i * deviceCount + j] << std::endl;
      }
    }
    std::cout << "p q in_feat:\n";
    std::cout << p << std::endl;
    std::cout << q << std::endl;
    std::cout << in_feat << std::endl;
    fflush(stdout);
  }

  // allocating GPU memory
  std::vector<float*> in_feat_s(deviceCount);
  std::vector<float*> out_feat_s(deviceCount);
  std::vector<float*> recv_buffer_s(deviceCount);
  std::vector<cudaStream_t> stream_s(deviceCount);
  std::vector<long> A_row_s(deviceCount);
  std::vector<long> A_col_s(deviceCount);
  std::vector<int*> d_row_ptr_s(deviceCount);
  std::vector<int*> d_col_idx_s(deviceCount);
  std::vector<float*> d_edge_val_s(deviceCount);

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamCreate(&stream_s[deviceId]));
  }

  long max_in_feat_size = 0;
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));

    long row_begin = p[deviceId].item<long>();
    long row_end = p[deviceId + 1].item<long>();
    long row_num = row_end - row_begin;
    A_row_s[deviceId] = row_num;
    long out_feat_size = row_num * f * sizeof(float);

    long col_begin = q[deviceId].item<long>();
    long col_end = q[deviceId + 1].item<long>();
    long col_num = col_end - col_begin;
    A_col_s[deviceId] = col_num;
    long in_feat_size = col_num * f * sizeof(float);
    max_in_feat_size = MAX(max_in_feat_size, in_feat_size);
    checkCudaError(cudaMallocAsync((void**)&in_feat_s[deviceId], in_feat_size,
                                   stream_s[deviceId]));
    checkCudaError(cudaMemcpyAsync(
        in_feat_s[deviceId], in_feat.data_ptr<float>() + col_begin * f,
        in_feat_size, cudaMemcpyHostToDevice, stream_s[deviceId]));
    checkCudaError(cudaMallocAsync((void**)&out_feat_s[deviceId], out_feat_size,
                                   stream_s[deviceId]));
  }

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaMallocAsync((void**)&recv_buffer_s[deviceId],
                                   max_in_feat_size, stream_s[deviceId]));
  }

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
  }

  // initializing NCCL
  ncclComm_t comms[4];
  int devs[4] = {0, 1, 2, 3};
  NCCLCHECK(ncclCommInitAll(comms, deviceCount, devs));

  if (MULTIGPU_SPMM_DEBUG) {
    std::cout << "finish initilizing NCCL\n";
    fflush(stdout);
  }

  // computing
  for (int stage = 0; stage < deviceCount; stage++) {
    if (MULTIGPU_SPMM_DEBUG) {
      std::cout << "stage " << stage << std::endl;
      fflush(stdout);
    }
    // broadcast B
    NCCLCHECK(ncclGroupStart());
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      NCCLCHECK(ncclBroadcast((const void*)in_feat_s[stage],
                              (void*)recv_buffer_s[deviceId],
                              A_col_s[stage] * f, ncclFloat, stage,
                              comms[deviceId], stream_s[deviceId]));
    }
    NCCLCHECK(ncclGroupEnd());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
    }

    if (MULTIGPU_SPMM_DEBUG) {
      std::cout << "finish broadcasting\n";
      fflush(stdout);
    }

    // copy A to GPU -> spmm -> free A
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      // copy A
      if (MULTIGPU_SPMM_DEBUG) {
        std::cout << deviceId << " start copy A\n";
        fflush(stdout);
      }
      auto& h_row_ptr = row_ptr_s[deviceId * deviceCount + stage];
      auto& h_col_idx = col_idx_s[deviceId * deviceCount + stage];
      auto& h_edge_val = edge_val_s[deviceId * deviceCount + stage];
      checkCudaError(cudaMallocAsync(&d_row_ptr_s[deviceId],
                                     h_row_ptr.size(0) * sizeof(int),
                                     stream_s[deviceId]));
      checkCudaError(cudaMallocAsync(&d_col_idx_s[deviceId],
                                     h_col_idx.size(0) * sizeof(int),
                                     stream_s[deviceId]));
      checkCudaError(cudaMallocAsync(&d_edge_val_s[deviceId],
                                     h_edge_val.size(0) * sizeof(float),
                                     stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_row_ptr_s[deviceId], h_row_ptr.data_ptr<int>(),
                          h_row_ptr.size(0) * sizeof(int),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_col_idx_s[deviceId], h_col_idx.data_ptr<int>(),
                          h_col_idx.size(0) * sizeof(int),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_edge_val_s[deviceId], h_edge_val.data_ptr<float>(),
                          h_edge_val.size(0) * sizeof(float),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      if (MULTIGPU_SPMM_DEBUG) {
        std::cout << deviceId << " finish copy A\n";
        fflush(stdout);
      }
      // spmm
      dim3 blockDim(32, (f + 31) / 32);
      dim3 gridDim(A_row_s[deviceId], 1);

      spmm_kernel<<<gridDim, blockDim, 0, stream_s[deviceId]>>>(
          f, d_row_ptr_s[deviceId], d_col_idx_s[deviceId],
          d_edge_val_s[deviceId], recv_buffer_s[deviceId], out_feat_s[deviceId],
          stage);

      if (MULTIGPU_SPMM_DEBUG) {
        std::cout << deviceId << " finish spmm\n";
        fflush(stdout);
      }
      // free A
      checkCudaError(cudaFreeAsync(d_row_ptr_s[deviceId], stream_s[deviceId]));
      checkCudaError(cudaFreeAsync(d_col_idx_s[deviceId], stream_s[deviceId]));
      checkCudaError(cudaFreeAsync(d_edge_val_s[deviceId], stream_s[deviceId]));
      if (MULTIGPU_SPMM_DEBUG) {
        std::cout << deviceId << " finish free A\n";
        fflush(stdout);
      }
    }

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
    }
  }

  // copy back
  if (MULTIGPU_SPMM_DEBUG) {
    std::cout << "start copy back\n";
    fflush(stdout);
  }
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaMemcpyAsync(
        out_feat.data_ptr<float>() + p[deviceId].item<long>() * f,
        out_feat_s[deviceId], A_row_s[deviceId] * f * sizeof(float),
        cudaMemcpyDeviceToHost, stream_s[deviceId]));
  }

  // free
  if (MULTIGPU_SPMM_DEBUG) {
    std::cout << "start free\n";
    fflush(stdout);
  }
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaFreeAsync(in_feat_s[deviceId], stream_s[deviceId]));
    checkCudaError(cudaFreeAsync(out_feat_s[deviceId], stream_s[deviceId]));
    checkCudaError(cudaFreeAsync(recv_buffer_s[deviceId], stream_s[deviceId]));
  }

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
    ncclCommDestroy(comms[deviceId]);

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
  }

  return out_feat;
}

torch::Tensor multigpu_gcn_inference_cuda(
    const long nnz, const int num_layers, const int num_hidden,
    const int num_classes,
    std::vector<torch::Tensor> row_ptr_s,   // int
    std::vector<torch::Tensor> col_idx_s,   // int
    std::vector<torch::Tensor> edge_val_s,  // float
    torch::Tensor p, torch::Tensor q,       // long
    torch::Tensor in_feat,                  // float
    std::vector<torch::Tensor> weight_s     // float
) {
  /*****************/
  /* initialization*/
  /*****************/
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  assert(deviceCount * deviceCount == row_ptr_s.size());
  assert(deviceCount * deviceCount == col_idx_s.size());
  assert(deviceCount * deviceCount == edge_val_s.size());
  assert(deviceCount + 1 == p.size(0));
  assert(deviceCount + 1 == q.size(0));
  const int m = in_feat.size(0);
  const int f = in_feat.size(1);
  int max_f = MAX(f, num_hidden);
  max_f = MAX(max_f, num_classes);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto out_feat = torch::empty({m, f}, options);

  // initializing NCCL
  ncclComm_t comms[4];
  int devs[4] = {0, 1, 2, 3};
  NCCLCHECK(ncclCommInitAll(comms, deviceCount, devs));

  if (MULTIGPU_SPMM_DEBUG) {
    std::cout << "finish initilizing NCCL\n";
    fflush(stdout);
  }

  // allocating GPU memory
  std::vector<float*> in_feat_s(deviceCount);
  std::vector<float*> out_feat_s(deviceCount);
  std::vector<float*> recv_buffer_s(deviceCount);
  std::vector<cudaStream_t> stream_s(deviceCount);
  std::vector<long> A_row_s(deviceCount);
  std::vector<long> A_col_s(deviceCount);
  std::vector<int*> d_row_ptr_s(deviceCount);
  std::vector<int*> d_col_idx_s(deviceCount);
  std::vector<float*> d_edge_val_s(deviceCount);

  // calculate max_col_num
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamCreate(&stream_s[deviceId]));
  }

  long max_col_num = 0;
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));

    long row_begin = p[deviceId].item<long>();
    long row_end = p[deviceId + 1].item<long>();
    long row_num = row_end - row_begin;
    A_row_s[deviceId] = row_num;
    // long out_feat_size = row_num * f * sizeof(float);

    long col_begin = q[deviceId].item<long>();
    long col_end = q[deviceId + 1].item<long>();
    long col_num = col_end - col_begin;
    A_col_s[deviceId] = col_num;
    max_col_num = MAX(max_col_num, col_num);
    // long in_feat_size = col_num * f * sizeof(float);
    // max_in_feat_size = MAX(max_in_feat_size, in_feat_size);

    // checkCudaError(cudaMemcpyAsync(
    //     in_feat_s[deviceId], in_feat.data_ptr<float>() + col_begin * f,
    //     in_feat_size, cudaMemcpyHostToDevice, stream_s[deviceId]));
  }

  long max_in_feat_size = max_col_num * max_f * sizeof(float);
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaMallocAsync((void**)&in_feat_s[deviceId],
                                   col_num * max_f * sizeof(float),
                                   stream_s[deviceId]));
    checkCudaError(cudaMallocAsync((void**)&out_feat_s[deviceId],
                                   row_num * max_f * sizeof(float),
                                   stream_s[deviceId]));
    checkCudaError(cudaMallocAsync((void**)&recv_buffer_s[deviceId],
                                   max_in_feat_size, stream_s[deviceId]));
  }

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
  }

  /*********************/
  // compute first layer
  /*********************/
  // copy in_feat to GPU
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    long in_feat_size = A_cols[deviceId] * f * sizeof(float);
    checkCudaError(cudaMemcpyAsync(
        in_feat_s[deviceId],
        in_feat.data_ptr<float>() + q[deviceId].item<long>() * f, in_feat_size,
        cudaMemcpyHostToDevice, stream_s[deviceId]));
  }

  // computing
  for (int stage = 0; stage < deviceCount; stage++) {
    if (MULTIGPU_SPMM_DEBUG) {
      std::cout << "stage " << stage << std::endl;
      fflush(stdout);
    }
    // broadcast B
    NCCLCHECK(ncclGroupStart());
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      NCCLCHECK(ncclBroadcast((const void*)in_feat_s[stage],
                              (void*)recv_buffer_s[deviceId],
                              A_col_s[stage] * f, ncclFloat, stage,
                              comms[deviceId], stream_s[deviceId]));
    }
    NCCLCHECK(ncclGroupEnd());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
    }

    if (MULTIGPU_SPMM_DEBUG) {
      std::cout << "finish broadcasting\n";
      fflush(stdout);
    }

    // copy A to GPU -> spmm -> free A
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      // copy A
      if (MULTIGPU_SPMM_DEBUG) {
        std::cout << deviceId << " start copy A\n";
        fflush(stdout);
      }
      auto& h_row_ptr = row_ptr_s[deviceId * deviceCount + stage];
      auto& h_col_idx = col_idx_s[deviceId * deviceCount + stage];
      auto& h_edge_val = edge_val_s[deviceId * deviceCount + stage];
      checkCudaError(cudaMallocAsync(&d_row_ptr_s[deviceId],
                                     h_row_ptr.size(0) * sizeof(int),
                                     stream_s[deviceId]));
      checkCudaError(cudaMallocAsync(&d_col_idx_s[deviceId],
                                     h_col_idx.size(0) * sizeof(int),
                                     stream_s[deviceId]));
      checkCudaError(cudaMallocAsync(&d_edge_val_s[deviceId],
                                     h_edge_val.size(0) * sizeof(float),
                                     stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_row_ptr_s[deviceId], h_row_ptr.data_ptr<int>(),
                          h_row_ptr.size(0) * sizeof(int),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_col_idx_s[deviceId], h_col_idx.data_ptr<int>(),
                          h_col_idx.size(0) * sizeof(int),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_edge_val_s[deviceId], h_edge_val.data_ptr<float>(),
                          h_edge_val.size(0) * sizeof(float),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      if (MULTIGPU_SPMM_DEBUG) {
        std::cout << deviceId << " finish copy A\n";
        fflush(stdout);
      }
      // spmm
      dim3 blockDim(32, (f + 31) / 32);
      dim3 gridDim(A_row_s[deviceId], 1);

      spmm_kernel<<<gridDim, blockDim, 0, stream_s[deviceId]>>>(
          f, d_row_ptr_s[deviceId], d_col_idx_s[deviceId],
          d_edge_val_s[deviceId], recv_buffer_s[deviceId], out_feat_s[deviceId],
          stage);

      if (MULTIGPU_SPMM_DEBUG) {
        std::cout << deviceId << " finish spmm\n";
        fflush(stdout);
      }
      // free A
      checkCudaError(cudaFreeAsync(d_row_ptr_s[deviceId], stream_s[deviceId]));
      checkCudaError(cudaFreeAsync(d_col_idx_s[deviceId], stream_s[deviceId]));
      checkCudaError(cudaFreeAsync(d_edge_val_s[deviceId], stream_s[deviceId]));
      if (MULTIGPU_SPMM_DEBUG) {
        std::cout << deviceId << " finish free A\n";
        fflush(stdout);
      }
    }

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
    }
  }

  // copy back
  if (MULTIGPU_SPMM_DEBUG) {
    std::cout << "start copy back\n";
    fflush(stdout);
  }
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaMemcpyAsync(
        out_feat.data_ptr<float>() + p[deviceId].item<long>() * f,
        out_feat_s[deviceId], A_row_s[deviceId] * f * sizeof(float),
        cudaMemcpyDeviceToHost, stream_s[deviceId]));
  }

  // free
  if (MULTIGPU_SPMM_DEBUG) {
    std::cout << "start free\n";
    fflush(stdout);
  }
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaFreeAsync(in_feat_s[deviceId], stream_s[deviceId]));
    checkCudaError(cudaFreeAsync(out_feat_s[deviceId], stream_s[deviceId]));
    checkCudaError(cudaFreeAsync(recv_buffer_s[deviceId], stream_s[deviceId]));
  }

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
    ncclCommDestroy(comms[deviceId]);

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
  }

  return out_feat;
}