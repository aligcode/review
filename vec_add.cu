#include <iostream>
#include <cuda_runtime.h>
using namespace std;


__global__ void vector_add(int* A, int* B, int* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


int main() {

    const int N = 10000;
    size_t d_vec_size = N * sizeof(int);

    // host arrays
    int* h_A = new int[N];
    int* h_B = new int[N];
    int* h_C = new int[N];


    for (int i=0; i < N; i++) {
        h_A[i] = rand() % 50;
        h_B[i] = rand() % 50;
    }

    // device arrays + allocate device memory
    int* d_A; 
    int* d_B;
    int* d_C;

    cudaMalloc(&d_A, d_vec_size);
    cudaMalloc(&d_B, d_vec_size);
    cudaMalloc(&d_C, d_vec_size);
    
    // transfer to device
    cudaMemcpy(d_A, h_A, d_vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, d_vec_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // copy back to host + print + free memory
    cudaMemcpy(h_C, d_C, d_vec_size, cudaMemcpyDeviceToHost);

    for(int i=0; i < N; i++) {
        cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}





