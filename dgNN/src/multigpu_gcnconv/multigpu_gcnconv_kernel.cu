#include <cublas_v2.h>
#include <cuda.h>
#include <nccl.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

#include "../util/computeUtil.h"

#define MULTIGPU_SPMM_DEBUG 0

__global__ void print_device_pointer_float(const float* ptr) {
  printf("print_device_pointer_float");
  for (int i = 0; i < 32; i++) {
    printf("%f ", ptr[i]);
  }
  printf("\n");
}

__global__ void start() { printf("start profiling\n"); }

__global__ void end() { printf("end profiling\n"); }

void startProfile() { start<<<1, 1>>>(); }
void endProfile() { end<<<1, 1>>>(); }

__global__ void relu_kernel(const long size, const float* in, float* out) {
  long idx = (long)blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = in[idx] > 0 ? in[idx] : 0;
  }
}

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

// relu kernel

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

void multigpu_gcnconv_inference_cuda(
    const int deviceCount, const ncclComm_t* comms, const int* devs,
    const cublasHandle_t* cublas_handle_s,      // handlers
    const std::vector<cudaStream_t>& stream_s,  // handlers
    const std::vector<float*>& in_feat_s,       // device pointer
    const std::vector<float*>& out_feat_s,      // device pointer
    const std::vector<float*>& recv_buffer_s,   // device pointer
    const std::vector<int*>& d_row_ptr_s,       // device pointer
    const std::vector<int*>& d_col_idx_s,       // device pointer
    const std::vector<float*>& d_edge_val_s,    // device pointer
    const std::vector<float*>& d_weight_s,      // device pointer
    const std::vector<long>& A_row_s, const std::vector<long>& A_col_s,
    const std::vector<long>& max_edge_num_s,
    const std::vector<torch::Tensor> row_ptr_s,   // int
    const std::vector<torch::Tensor> col_idx_s,   // int
    const std::vector<torch::Tensor> edge_val_s,  // float
    const float* weight,                          // float
    const int in_feat_size, const int out_feat_size) {
  // TODO: operator reorder
  if (out_feat_size > in_feat_size) {
    const int f = in_feat_size;
    // spmm
    for (int stage = 0; stage < deviceCount; stage++) {
      // broadcast B
      NCCLCHECK(ncclGroupStart());
      for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        NCCLCHECK(ncclBroadcast((const void*)in_feat_s[stage],
                                (void*)recv_buffer_s[deviceId],
                                A_col_s[stage] * f, ncclFloat, stage,
                                comms[deviceId], stream_s[deviceId]));
      }
      NCCLCHECK(ncclGroupEnd());

      // copy A to GPU -> spmm -> free A
      for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        checkCudaError(cudaSetDevice(deviceId));
        // copy A
        auto& h_row_ptr = row_ptr_s[deviceId * deviceCount + stage];
        auto& h_col_idx = col_idx_s[deviceId * deviceCount + stage];
        auto& h_edge_val = edge_val_s[deviceId * deviceCount + stage];
        checkCudaError(
            cudaMemcpyAsync(d_row_ptr_s[deviceId], h_row_ptr.data_ptr<int>(),
                            h_row_ptr.size(0) * sizeof(int),
                            cudaMemcpyHostToDevice, stream_s[deviceId]));
        checkCudaError(
            cudaMemcpyAsync(d_col_idx_s[deviceId], h_col_idx.data_ptr<int>(),
                            h_col_idx.size(0) * sizeof(int),
                            cudaMemcpyHostToDevice, stream_s[deviceId]));
        checkCudaError(cudaMemcpyAsync(
            d_edge_val_s[deviceId], h_edge_val.data_ptr<float>(),
            h_edge_val.size(0) * sizeof(float), cudaMemcpyHostToDevice,
            stream_s[deviceId]));

        // spmm
        dim3 blockDim(32, (f + 31) / 32);
        dim3 gridDim(A_row_s[deviceId], 1);

        spmm_kernel<<<gridDim, blockDim, 0, stream_s[deviceId]>>>(
            f, d_row_ptr_s[deviceId], d_col_idx_s[deviceId],
            d_edge_val_s[deviceId], recv_buffer_s[deviceId],
            out_feat_s[deviceId], stage);
      }
    }
    // result is now in out_feat_s

    // gemm
    float alpha = 1.0;
    float beta = 0.0;
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCublasError(
          cublasSetStream(cublas_handle_s[deviceId], stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_weight_s[deviceId], weight,
                          in_feat_size * out_feat_size * sizeof(float),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      int m = A_row_s[deviceId];
      int n = out_feat_size;
      int k = in_feat_size;
      checkCublasError(
          cublasSgemm(cublas_handle_s[deviceId], CUBLAS_OP_N, CUBLAS_OP_N, n, m,
                      k, &alpha, d_weight_s[deviceId], n, out_feat_s[deviceId],
                      k, &beta, in_feat_s[deviceId], n));
      // result is now in in_feat_s
      long array_size = long(m) * long(n);
      relu_kernel<<<(array_size + 1023) / 1024, 1024, 0, stream_s[deviceId]>>>(
          array_size, in_feat_s[deviceId], out_feat_s[deviceId]);
      // result is now in out_feat_s
    }

  } else {
    // gemm
    float alpha = 1.0;
    float beta = 0.0;
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCublasError(
          cublasSetStream(cublas_handle_s[deviceId], stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_weight_s[deviceId], weight,
                          in_feat_size * out_feat_size * sizeof(float),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      int m = A_row_s[deviceId];
      int n = out_feat_size;
      int k = in_feat_size;
      checkCublasError(cublasSgemm(cublas_handle_s[deviceId], CUBLAS_OP_N,
                                   CUBLAS_OP_N, n, m, k, &alpha,
                                   d_weight_s[deviceId], n, in_feat_s[deviceId],
                                   k, &beta, out_feat_s[deviceId], n));
    }
    // result is now in out_feat_s

    const int f = out_feat_size;
    // spmm
    for (int stage = 0; stage < deviceCount; stage++) {
      // broadcast B
      NCCLCHECK(ncclGroupStart());
      for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        NCCLCHECK(ncclBroadcast((const void*)out_feat_s[stage],
                                (void*)recv_buffer_s[deviceId],
                                A_col_s[stage] * f, ncclFloat, stage,
                                comms[deviceId], stream_s[deviceId]));
      }
      NCCLCHECK(ncclGroupEnd());

      // copy A to GPU -> spmm -> free A
      for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        checkCudaError(cudaSetDevice(deviceId));
        // copy A
        auto& h_row_ptr = row_ptr_s[deviceId * deviceCount + stage];
        auto& h_col_idx = col_idx_s[deviceId * deviceCount + stage];
        auto& h_edge_val = edge_val_s[deviceId * deviceCount + stage];
        checkCudaError(
            cudaMemcpyAsync(d_row_ptr_s[deviceId], h_row_ptr.data_ptr<int>(),
                            h_row_ptr.size(0) * sizeof(int),
                            cudaMemcpyHostToDevice, stream_s[deviceId]));
        checkCudaError(
            cudaMemcpyAsync(d_col_idx_s[deviceId], h_col_idx.data_ptr<int>(),
                            h_col_idx.size(0) * sizeof(int),
                            cudaMemcpyHostToDevice, stream_s[deviceId]));
        checkCudaError(cudaMemcpyAsync(
            d_edge_val_s[deviceId], h_edge_val.data_ptr<float>(),
            h_edge_val.size(0) * sizeof(float), cudaMemcpyHostToDevice,
            stream_s[deviceId]));

        // spmm
        dim3 blockDim(32, (f + 31) / 32);
        dim3 gridDim(A_row_s[deviceId], 1);

        spmm_kernel<<<gridDim, blockDim, 0, stream_s[deviceId]>>>(
            f, d_row_ptr_s[deviceId], d_col_idx_s[deviceId],
            d_edge_val_s[deviceId], recv_buffer_s[deviceId],
            in_feat_s[deviceId], stage);
      }
    }

    // relu
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      int m = A_row_s[deviceId];
      int n = out_feat_size;
      long array_size = long(m) * long(n);
      relu_kernel<<<(array_size + 1023) / 1024, 1024, 0, stream_s[deviceId]>>>(
          array_size, in_feat_s[deviceId], out_feat_s[deviceId]);
    }
  }
}

std::vector<torch::Tensor> multigpu_gcn_inference_cuda(
    const long nnz, const int num_layers, const int num_hidden,
    const int num_classes,
    const std::vector<torch::Tensor> row_ptr_s,    // int
    const std::vector<torch::Tensor> col_idx_s,    // int
    const std::vector<torch::Tensor> edge_val_s,   // float
    const torch::Tensor p, const torch::Tensor q,  // long
    const torch::Tensor in_feat,                   // float
    const std::vector<torch::Tensor> weight_s      // float
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
  assert(weight_s.size() == num_layers);
  assert(weight_s[0].size(0) == in_feat.size(1));
  assert(weight_s[0].size(1) == num_hidden);
  for (int i = 1; i < num_layers - 1; i++) {
    assert(weight_s[i].size(0) == num_hidden);
    assert(weight_s[i].size(1) == num_hidden);
  }
  assert(weight_s[num_layers - 1].size(0) == num_hidden);
  assert(weight_s[num_layers - 1].size(1) == num_classes);
  const int m = in_feat.size(0);
  const int f = in_feat.size(1);
  int max_f = MAX(f, num_hidden);
  max_f = MAX(max_f, num_classes);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto out_feat = torch::empty({m, num_classes}, options);
  auto inference_time = torch::empty({10}, options);

  // initialize
  ncclComm_t comms[4];
  int devs[4] = {0, 1, 2, 3};
  NCCLCHECK(ncclCommInitAll(comms, deviceCount, devs));

  cublasHandle_t cublas_handle_s[4];
  for (int i = 0; i < deviceCount; i++) {
    checkCudaError(cudaSetDevice(i));
    checkCublasError(cublasCreate(&cublas_handle_s[i]));
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
  std::vector<long> max_edge_num_s(deviceCount);
  std::vector<float*> d_weight_s(deviceCount);

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamCreate(&stream_s[deviceId]));
  }

  long max_col_num = 0;
  long max_row_num = 0;
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));

    long row_begin = p[deviceId].item<long>();
    long row_end = p[deviceId + 1].item<long>();
    long row_num = row_end - row_begin;
    A_row_s[deviceId] = row_num;
    max_row_num = MAX(max_row_num, row_num);

    long col_begin = q[deviceId].item<long>();
    long col_end = q[deviceId + 1].item<long>();
    long col_num = col_end - col_begin;
    A_col_s[deviceId] = col_num;
    max_col_num = MAX(max_col_num, col_num);

    long max_edge_num = 0;

    for (int i = 0; i < deviceCount; i++) {
      max_edge_num =
          MAX(max_edge_num,
              row_ptr_s[deviceId * deviceCount + i][row_num].item<long>());
    }
    max_edge_num_s[deviceId] = max_edge_num;
  }
  long max_row_col_num = MAX(max_row_num, max_col_num);

  long max_feat_size = max_row_col_num * max_f * sizeof(float);
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    long row_num = A_row_s[deviceId];
    long col_num = A_col_s[deviceId];
    checkCudaError(cudaMallocAsync((void**)&in_feat_s[deviceId], max_feat_size,
                                   stream_s[deviceId]));
    checkCudaError(cudaMallocAsync((void**)&out_feat_s[deviceId], max_feat_size,
                                   stream_s[deviceId]));
    checkCudaError(cudaMallocAsync((void**)&recv_buffer_s[deviceId],
                                   max_feat_size, stream_s[deviceId]));
    checkCudaError(cudaMallocAsync(&d_row_ptr_s[deviceId],
                                   (row_num + 1) * sizeof(int),
                                   stream_s[deviceId]));
    checkCudaError(cudaMallocAsync(&d_col_idx_s[deviceId],
                                   max_edge_num_s[deviceId] * sizeof(int),
                                   stream_s[deviceId]));
    checkCudaError(cudaMallocAsync(&d_edge_val_s[deviceId],
                                   max_edge_num_s[deviceId] * sizeof(float),
                                   stream_s[deviceId]));
    checkCudaError(cudaMallocAsync(&d_weight_s[deviceId],
                                   max_f * max_f * sizeof(float),
                                   stream_s[deviceId]));
    checkCudaError(cudaMemcpyAsync(
        in_feat_s[deviceId],
        in_feat.data_ptr<float>() + q[deviceId].item<long>() * f,
        col_num * f * sizeof(float), cudaMemcpyHostToDevice,
        stream_s[deviceId]));
  }

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
  }

  for (int epoch = 0; epoch < 20; epoch++) {
    // copy in_feat to GPU
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      long col_num = A_col_s[deviceId];
      checkCudaError(cudaMemcpyAsync(
          in_feat_s[deviceId],
          in_feat.data_ptr<float>() + q[deviceId].item<long>() * f,
          col_num * f * sizeof(float), cudaMemcpyHostToDevice,
          stream_s[deviceId]));
    }
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
    }
    startProfile();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // first layer
    multigpu_gcnconv_inference_cuda(
        deviceCount, comms, devs, cublas_handle_s, stream_s, in_feat_s,
        out_feat_s, recv_buffer_s, d_row_ptr_s, d_col_idx_s, d_edge_val_s,
        d_weight_s, A_row_s, A_col_s, max_edge_num_s, row_ptr_s, col_idx_s,
        edge_val_s, weight_s[0].data_ptr<float>(), weight_s[0].size(0),
        weight_s[0].size(1));
    // result in out_feat_s

    // hidden layers
    for (int i = 1; i < num_layers - 1; i++) {
      std::swap(in_feat_s, out_feat_s);
      multigpu_gcnconv_inference_cuda(
          deviceCount, comms, devs, cublas_handle_s, stream_s, in_feat_s,
          out_feat_s, recv_buffer_s, d_row_ptr_s, d_col_idx_s, d_edge_val_s,
          d_weight_s, A_row_s, A_col_s, max_edge_num_s, row_ptr_s, col_idx_s,
          edge_val_s, weight_s[i].data_ptr<float>(), weight_s[i].size(0),
          weight_s[i].size(1));
    }

    // last layer
    std::swap(in_feat_s, out_feat_s);
    multigpu_gcnconv_inference_cuda(
        deviceCount, comms, devs, cublas_handle_s, stream_s, in_feat_s,
        out_feat_s, recv_buffer_s, d_row_ptr_s, d_col_idx_s, d_edge_val_s,
        d_weight_s, A_row_s, A_col_s, max_edge_num_s, row_ptr_s, col_idx_s,
        edge_val_s, weight_s[num_layers - 1].data_ptr<float>(),
        weight_s[num_layers - 1].size(0), weight_s[num_layers - 1].size(1));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time on GPU "
              << ": " << elapsedTime << " ms" << std::endl;
    if (epoch >= 10) inference_time[epoch - 10] = elapsedTime;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    fflush(stdout);
    endProfile();
  }

  // copy back
  if (MULTIGPU_SPMM_DEBUG) {
    std::cout << "start copy back\n";
    fflush(stdout);
  }
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaMemcpyAsync(
        out_feat.data_ptr<float>() + p[deviceId].item<long>() * num_classes,
        out_feat_s[deviceId], A_row_s[deviceId] * num_classes * sizeof(float),
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
    checkCudaError(cudaFreeAsync(d_row_ptr_s[deviceId], stream_s[deviceId]));
    checkCudaError(cudaFreeAsync(d_col_idx_s[deviceId], stream_s[deviceId]));
    checkCudaError(cudaFreeAsync(d_edge_val_s[deviceId], stream_s[deviceId]));
    checkCudaError(cudaFreeAsync(d_weight_s[deviceId], stream_s[deviceId]));
  }

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
    ncclCommDestroy(comms[deviceId]);
    cublasDestroy(cublas_handle_s[deviceId]);
  }

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
  }

  return {out_feat, inference_time};
}

void multigpu_gcnconv_inference_unified_memory_cuda(
    const int deviceCount, const ncclComm_t* comms, const int* devs,
    const cublasHandle_t* cublas_handle_s,      // handlers
    const std::vector<cudaStream_t>& stream_s,  // handlers
    const std::vector<float*>& in_feat_s,       // device pointer
    const std::vector<float*>& out_feat_s,      // device pointer
    const std::vector<float*>& recv_buffer_s,   // device pointer
    const std::vector<int*>& um_row_ptr_s,      // unified memory pointer
    const std::vector<int*>& um_col_idx_s,      // unified pointer
    const std::vector<float*>& um_edge_val_s,   // unified pointer
    const std::vector<float*>& d_weight_s,      // device pointer
    const std::vector<long>& A_row_s, const std::vector<long>& A_col_s,
    const std::vector<long>& max_edge_num_s,
    const std::vector<torch::Tensor> row_ptr_s,   // int
    const std::vector<torch::Tensor> col_idx_s,   // int
    const std::vector<torch::Tensor> edge_val_s,  // float
    const float* weight,                          // float
    const int in_feat_size, const int out_feat_size) {
  // TODO: operator reorder
  if (out_feat_size > in_feat_size) {
    const int f = in_feat_size;
    // spmm
    for (int stage = 0; stage < deviceCount; stage++) {
      // broadcast B
      NCCLCHECK(ncclGroupStart());
      for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        NCCLCHECK(ncclBroadcast((const void*)in_feat_s[stage],
                                (void*)recv_buffer_s[deviceId],
                                A_col_s[stage] * f, ncclFloat, stage,
                                comms[deviceId], stream_s[deviceId]));
      }
      NCCLCHECK(ncclGroupEnd());

      // spmm
      for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        checkCudaError(cudaSetDevice(deviceId));
        int index = deviceId * deviceCount + stage;

        dim3 blockDim(32, (f + 31) / 32);
        dim3 gridDim(A_row_s[deviceId], 1);

        spmm_kernel<<<gridDim, blockDim, 0, stream_s[deviceId]>>>(
            f, um_row_ptr_s[index], um_col_idx_s[index], um_edge_val_s[index],
            recv_buffer_s[deviceId], out_feat_s[deviceId], stage);
      }
    }
    // result is now in out_feat_s

    // gemm
    float alpha = 1.0;
    float beta = 0.0;
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCublasError(
          cublasSetStream(cublas_handle_s[deviceId], stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_weight_s[deviceId], weight,
                          in_feat_size * out_feat_size * sizeof(float),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      int m = A_row_s[deviceId];
      int n = out_feat_size;
      int k = in_feat_size;
      checkCublasError(
          cublasSgemm(cublas_handle_s[deviceId], CUBLAS_OP_N, CUBLAS_OP_N, n, m,
                      k, &alpha, d_weight_s[deviceId], n, out_feat_s[deviceId],
                      k, &beta, in_feat_s[deviceId], n));
      // result is now in in_feat_s
      long array_size = long(m) * long(n);
      relu_kernel<<<(array_size + 1023) / 1024, 1024, 0, stream_s[deviceId]>>>(
          array_size, in_feat_s[deviceId], out_feat_s[deviceId]);
      // result is now in out_feat_s
    }

  } else {
    // gemm
    float alpha = 1.0;
    float beta = 0.0;
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCublasError(
          cublasSetStream(cublas_handle_s[deviceId], stream_s[deviceId]));
      checkCudaError(
          cudaMemcpyAsync(d_weight_s[deviceId], weight,
                          in_feat_size * out_feat_size * sizeof(float),
                          cudaMemcpyHostToDevice, stream_s[deviceId]));
      int m = A_row_s[deviceId];
      int n = out_feat_size;
      int k = in_feat_size;
      checkCublasError(cublasSgemm(cublas_handle_s[deviceId], CUBLAS_OP_N,
                                   CUBLAS_OP_N, n, m, k, &alpha,
                                   d_weight_s[deviceId], n, in_feat_s[deviceId],
                                   k, &beta, out_feat_s[deviceId], n));
    }
    // result is now in out_feat_s

    const int f = out_feat_size;
    // spmm
    for (int stage = 0; stage < deviceCount; stage++) {
      // broadcast B
      NCCLCHECK(ncclGroupStart());
      for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        NCCLCHECK(ncclBroadcast((const void*)out_feat_s[stage],
                                (void*)recv_buffer_s[deviceId],
                                A_col_s[stage] * f, ncclFloat, stage,
                                comms[deviceId], stream_s[deviceId]));
      }
      NCCLCHECK(ncclGroupEnd());

      // copy A to GPU -> spmm -> free A
      for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        checkCudaError(cudaSetDevice(deviceId));
        int index = deviceId * deviceCount + stage;

        // spmm
        dim3 blockDim(32, (f + 31) / 32);
        dim3 gridDim(A_row_s[deviceId], 1);

        spmm_kernel<<<gridDim, blockDim, 0, stream_s[deviceId]>>>(
            f, um_row_ptr_s[index], um_col_idx_s[index], um_edge_val_s[index],
            recv_buffer_s[deviceId], in_feat_s[deviceId], stage);
      }
    }

    // relu
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      int m = A_row_s[deviceId];
      int n = out_feat_size;
      long array_size = long(m) * long(n);
      relu_kernel<<<(array_size + 1023) / 1024, 1024, 0, stream_s[deviceId]>>>(
          array_size, in_feat_s[deviceId], out_feat_s[deviceId]);
    }
  }
}

std::vector<torch::Tensor> multigpu_gcn_inference_unified_memory_cuda(
    const long nnz, const int num_layers, const int num_hidden,
    const int num_classes,
    const std::vector<torch::Tensor> row_ptr_s,    // int
    const std::vector<torch::Tensor> col_idx_s,    // int
    const std::vector<torch::Tensor> edge_val_s,   // float
    const torch::Tensor p, const torch::Tensor q,  // long
    const torch::Tensor in_feat,                   // float
    const std::vector<torch::Tensor> weight_s      // float
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
  assert(weight_s.size() == num_layers);
  assert(weight_s[0].size(0) == in_feat.size(1));
  assert(weight_s[0].size(1) == num_hidden);
  for (int i = 1; i < num_layers - 1; i++) {
    assert(weight_s[i].size(0) == num_hidden);
    assert(weight_s[i].size(1) == num_hidden);
  }
  assert(weight_s[num_layers - 1].size(0) == num_hidden);
  assert(weight_s[num_layers - 1].size(1) == num_classes);
  const int m = in_feat.size(0);
  const int f = in_feat.size(1);
  int max_f = MAX(f, num_hidden);
  max_f = MAX(max_f, num_classes);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto out_feat = torch::empty({m, num_classes}, options);
  auto inference_time = torch::empty({10}, options);

  // initialize
  ncclComm_t comms[4];
  int devs[4] = {0, 1, 2, 3};
  NCCLCHECK(ncclCommInitAll(comms, deviceCount, devs));

  cublasHandle_t cublas_handle_s[4];
  for (int i = 0; i < deviceCount; i++) {
    checkCudaError(cudaSetDevice(i));
    checkCublasError(cublasCreate(&cublas_handle_s[i]));
  }

  // allocating GPU memory
  std::vector<float*> in_feat_s(deviceCount);
  std::vector<float*> out_feat_s(deviceCount);
  std::vector<float*> recv_buffer_s(deviceCount);
  std::vector<cudaStream_t> stream_s(deviceCount);
  std::vector<long> A_row_s(deviceCount);
  std::vector<long> A_col_s(deviceCount);
  std::vector<int*> um_row_ptr_s(deviceCount * deviceCount);
  std::vector<int*> um_col_idx_s(deviceCount * deviceCount);
  std::vector<float*> um_edge_val_s(deviceCount * deviceCount);
  std::vector<long> max_edge_num_s(deviceCount);
  std::vector<float*> d_weight_s(deviceCount);

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamCreate(&stream_s[deviceId]));
  }

  long max_col_num = 0;
  long max_row_num = 0;
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));

    long row_begin = p[deviceId].item<long>();
    long row_end = p[deviceId + 1].item<long>();
    long row_num = row_end - row_begin;
    A_row_s[deviceId] = row_num;
    max_row_num = MAX(max_row_num, row_num);

    long col_begin = q[deviceId].item<long>();
    long col_end = q[deviceId + 1].item<long>();
    long col_num = col_end - col_begin;
    A_col_s[deviceId] = col_num;
    max_col_num = MAX(max_col_num, col_num);

    long max_edge_num = 0;

    for (int i = 0; i < deviceCount; i++) {
      max_edge_num =
          MAX(max_edge_num,
              row_ptr_s[deviceId * deviceCount + i][row_num].item<long>());
    }
    max_edge_num_s[deviceId] = max_edge_num;
  }
  long max_row_col_num = MAX(max_row_num, max_col_num);

  long max_feat_size = max_row_col_num * max_f * sizeof(float);
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    long row_num = A_row_s[deviceId];
    long col_num = A_col_s[deviceId];
    checkCudaError(cudaMallocAsync((void**)&in_feat_s[deviceId], max_feat_size,
                                   stream_s[deviceId]));
    checkCudaError(cudaMallocAsync((void**)&out_feat_s[deviceId], max_feat_size,
                                   stream_s[deviceId]));
    checkCudaError(cudaMallocAsync((void**)&recv_buffer_s[deviceId],
                                   max_feat_size, stream_s[deviceId]));
    // checkCudaError(cudaMallocAsync(&d_row_ptr_s[deviceId],
    //                                (row_num + 1) * sizeof(int),
    //                                stream_s[deviceId]));
    // checkCudaError(cudaMallocAsync(&d_col_idx_s[deviceId],
    //                                max_edge_num_s[deviceId] * sizeof(int),
    //                                stream_s[deviceId]));
    // checkCudaError(cudaMallocAsync(&d_edge_val_s[deviceId],
    //                                max_edge_num_s[deviceId] * sizeof(float),
    //                                stream_s[deviceId]));
    checkCudaError(cudaMallocAsync(&d_weight_s[deviceId],
                                   max_f * max_f * sizeof(float),
                                   stream_s[deviceId]));
    checkCudaError(cudaMemcpyAsync(
        in_feat_s[deviceId],
        in_feat.data_ptr<float>() + q[deviceId].item<long>() * f,
        col_num * f * sizeof(float), cudaMemcpyHostToDevice,
        stream_s[deviceId]));
  }
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    long row_num = A_row_s[deviceId];
    for (int stage = 0; stage < deviceCount; stage++) {
      int index = deviceId * deviceCount + stage;
      long edge_num = row_ptr_s[index][row_num].item<long>();
      checkCudaError(
          cudaMallocManaged(&um_row_ptr_s[index], (row_num + 1) * sizeof(int)));
      checkCudaError(
          cudaMallocManaged(&um_col_idx_s[index], edge_num * sizeof(int)));
      checkCudaError(
          cudaMallocManaged(&um_edge_val_s[index], edge_num * sizeof(float)));
      auto& h_row_ptr = row_ptr_s[index];
      auto& h_col_idx = col_idx_s[index];
      auto& h_edge_val = edge_val_s[index];
      checkCudaError(cudaMemcpy(um_row_ptr_s[index], h_row_ptr.data_ptr<int>(),
                                h_row_ptr.size(0) * sizeof(int),
                                cudaMemcpyDefault));
      checkCudaError(cudaMemcpy(um_col_idx_s[index], h_col_idx.data_ptr<int>(),
                                h_col_idx.size(0) * sizeof(int),
                                cudaMemcpyDefault));
      checkCudaError(
          cudaMemcpy(um_edge_val_s[index], h_edge_val.data_ptr<float>(),
                     h_edge_val.size(0) * sizeof(float), cudaMemcpyDefault));
    }
  }

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
    checkCudaError(cudaDeviceSynchronize());
  }

  for (int epoch = 0; epoch < 20; epoch++) {
    // copy in_feat to GPU
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      long col_num = A_col_s[deviceId];
      checkCudaError(cudaMemcpyAsync(
          in_feat_s[deviceId],
          in_feat.data_ptr<float>() + q[deviceId].item<long>() * f,
          col_num * f * sizeof(float), cudaMemcpyHostToDevice,
          stream_s[deviceId]));
    }
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      checkCudaError(cudaSetDevice(deviceId));
      checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
    }
    startProfile();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // first layer
    multigpu_gcnconv_inference_unified_memory_cuda(
        deviceCount, comms, devs, cublas_handle_s, stream_s, in_feat_s,
        out_feat_s, recv_buffer_s, um_row_ptr_s, um_col_idx_s, um_edge_val_s,
        d_weight_s, A_row_s, A_col_s, max_edge_num_s, row_ptr_s, col_idx_s,
        edge_val_s, weight_s[0].data_ptr<float>(), weight_s[0].size(0),
        weight_s[0].size(1));
    // result in out_feat_s

    // hidden layers
    for (int i = 1; i < num_layers - 1; i++) {
      std::swap(in_feat_s, out_feat_s);
      multigpu_gcnconv_inference_unified_memory_cuda(
          deviceCount, comms, devs, cublas_handle_s, stream_s, in_feat_s,
          out_feat_s, recv_buffer_s, um_row_ptr_s, um_col_idx_s, um_edge_val_s,
          d_weight_s, A_row_s, A_col_s, max_edge_num_s, row_ptr_s, col_idx_s,
          edge_val_s, weight_s[i].data_ptr<float>(), weight_s[i].size(0),
          weight_s[i].size(1));
    }

    // last layer
    std::swap(in_feat_s, out_feat_s);
    multigpu_gcnconv_inference_unified_memory_cuda(
        deviceCount, comms, devs, cublas_handle_s, stream_s, in_feat_s,
        out_feat_s, recv_buffer_s, um_row_ptr_s, um_col_idx_s, um_edge_val_s,
        d_weight_s, A_row_s, A_col_s, max_edge_num_s, row_ptr_s, col_idx_s,
        edge_val_s, weight_s[num_layers - 1].data_ptr<float>(),
        weight_s[num_layers - 1].size(0), weight_s[num_layers - 1].size(1));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time on GPU "
              << ": " << elapsedTime << " ms" << std::endl;
    if (epoch >= 10) inference_time[epoch - 10] = elapsedTime;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    fflush(stdout);
    endProfile();
  }

  // copy back
  if (MULTIGPU_SPMM_DEBUG) {
    std::cout << "start copy back\n";
    fflush(stdout);
  }
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaMemcpyAsync(
        out_feat.data_ptr<float>() + p[deviceId].item<long>() * num_classes,
        out_feat_s[deviceId], A_row_s[deviceId] * num_classes * sizeof(float),
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
    checkCudaError(cudaFreeAsync(d_weight_s[deviceId], stream_s[deviceId]));
  }
  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    for (int stage = 0; stage < deviceCount; stage++) {
      int index = deviceId * deviceCount + stage;
      checkCudaError(cudaFree(um_row_ptr_s[index]));
      checkCudaError(cudaFree(um_col_idx_s[index]));
      checkCudaError(cudaFree(um_edge_val_s[index]));
    }
  }

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
    ncclCommDestroy(comms[deviceId]);
    cublasDestroy(cublas_handle_s[deviceId]);
  }

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    checkCudaError(cudaSetDevice(deviceId));
    checkCudaError(cudaStreamSynchronize(stream_s[deviceId]));
  }

  return {out_feat, inference_time};
}
