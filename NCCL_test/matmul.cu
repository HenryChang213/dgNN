#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
// #include <nccl.h>
#include <fstream>

#define CUDA_CHECK(func)                                                         \
    {                                                                            \
        cudaError_t status = (func);                                             \
        if (status != cudaSuccess)                                               \
        {                                                                        \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
                   cudaGetErrorString(status), status);                          \
            return EXIT_FAILURE;                                                 \
        }                                                                        \
    }

// Thread block size
#define BLOCK_SIZE 32

__global__ void matrixMultiply(const double *A, const double *B, double *C, long long N) {
    long long row = blockIdx.y * blockDim.y + threadIdx.y;
    long long col = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0;
    for (int i = 0; i < N; ++i) {
        value += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = value;
}

void print_matrix(float *matrix, int N) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", matrix[i*N+j]);
        }

        printf("\n");
    }
}

void read_binary_2d_array(double* array, const std::string &filename, long long N){
    std::ifstream file(filename, std::ios::binary);
    for (long long i = 0; i < N; i++)
        for (long long j = 0; j < N;j++)
            file.read(reinterpret_cast<char *>(&array[i * N + j]), sizeof(double));
}

int main(int argc, char *argv[]) {
    long long N = 65536; // Matrix size
    size_t matrixSize = N * N * sizeof(double);

    // Initialize NCCL
    // ncclComm_t comm;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("device count=%d, matrix size=%ld(%ldGB)\n", deviceCount, N, matrixSize / 1024 / 1024 / 1024);
    fflush(stdout);

    // ncclUniqueId id;
    // ncclGetUniqueId(&id);
    // ncclCommInitAll(&comm, deviceCount, (const int*)&id);

    // Allocate memory on both GPUs
    double *h_A, *h_B, *h_C;
    h_A = (double *)malloc(matrixSize);
    h_B = h_A;
    h_C = (double *)malloc(matrixSize);

    read_binary_2d_array(h_A, "/home/hz0567/data/test/random_65536.bin", N);
    printf("finish reading\n");
    fflush(stdout);

    // CUDA_CHECK(cudaMallocHost((void **)&h_A, matrixSize));
    // CUDA_CHECK(cudaMallocHost((void**)&h_B, matrixSize));
    // CUDA_CHECK(cudaMallocHost((void**)&h_C, matrixSize));

    // Initialize matrices
    // for (long long i = 0; i < N * N; ++i)
    // {
    //     h_A[i] = std::rand() / (float)RAND_MAX;
    //     h_B[i] = std::rand() / (float)RAND_MAX;
    //     // h_B[i] = 1;
    // }
    double *d_A[4];
    double *d_B[4];
    double *d_C[4];

    for (int gpu = 0; gpu < deviceCount; ++gpu)
    {

        cudaSetDevice(gpu);

        cudaMalloc((void **)&d_A[gpu], matrixSize / deviceCount);
        cudaMalloc((void **)&d_B[gpu], matrixSize);
        cudaMalloc((void **)&d_C[gpu], matrixSize / deviceCount);

        cudaMemcpyAsync(d_A[gpu], h_A + gpu * (N * N / deviceCount), matrixSize / deviceCount, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_B[gpu], h_B, matrixSize, cudaMemcpyHostToDevice);

    }
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N / deviceCount + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < 1;i++){

        // Perform matrix multiplication on each GPU
        for (int gpu = 0; gpu < deviceCount; ++gpu)
        {
            cudaSetDevice(gpu);

            dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N / deviceCount + blockDim.y - 1) / blockDim.y);
            matrixMultiply<<<gridDim, blockDim>>>(d_A[gpu], d_B[gpu], d_C[gpu], N);


        }

    
    }
    for (int gpu = 0; gpu < deviceCount;gpu++){
        cudaSetDevice(gpu);
        cudaMemcpyAsync(h_C + gpu * (N * N / deviceCount), d_C[gpu], matrixSize / deviceCount, cudaMemcpyDeviceToHost);
    }
    // Record the stop event and measure the elapsed time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the elapsed time in milliseconds for the current GPU
    std::cout << "Elapsed time on GPU " << ": " << elapsedTime << " ms" << std::endl;

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Beginning checking\n");
    // Compute ground truth
    int wrong_num = 0;
    float *h_C_gt = (float *)malloc(matrixSize);
    for (long long i = 0; i < N; i=i+100) {
        for (long long j = 0 ; j < N ; j=j+100) {
            h_C_gt[ i * N + j ] = 0;
            for (int k = 0; k < N  ; k++) {
                h_C_gt [ i * N + j ] += h_A[ i * N + k ] * h_B[ k * N + j ];

            }

            float diff = h_C_gt[ i * N +j ]- h_C[ i * N + j ];
            if (diff > 1e-2 || diff < -1e-2) {
                printf("Wrong result %d,%d,%f\n",i,j,diff);
                wrong_num++;
                if(wrong_num > 10)
                    exit(-1);
            }
        }
    }

    // printf("Matrix A\n");
    // print_matrix(h_A, N);
    // printf("Matrix B\n");
    // print_matrix(h_B, N);
    // printf("Matrix C\n");
    // print_matrix(h_C, N);
    // printf("Matrix C_gt\n");
    // print_matrix(h_C_gt, N);



    // Finalize NCCL
    // ncclCommDestroy(comm);

    // Free host memory
    // cudaFreeHost(h_A);
    // cudaFreeHost(h_B);
    // cudaFreeHost(h_C);

    return 0;
}
