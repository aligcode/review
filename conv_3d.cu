#include <cuda_runtime.h>
#include <iostream>


using namespace std;

__global__ void convolution3d(float* image, float* kernel, float* output, int image_width, int image_height, int num_channels, int kernel_size) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // TODO: implement the kernel with the xyz approach
    if (x < image_width && y < image_height && z < num_channels) {
        
    }
    
}




int main() {

    // 1024 * 1024 * 3 image
    int image_width = 1024;
    int image_height = 1024;
    int num_channels = 3;
    
    // 3 * 3 * 3 kernel
    int kernel_size = 3;

    float* h_image = new float[image_width * image_height * num_channels];
    float* h_kernel = new float[kernel_size * kernel_size * kernel_size];
    float* h_output = new float[image_width * image_height * num_channels];

    for (int i = 0; i < image_width * image_heigth * num_channels; i++) {
        h_image[i] = rand() % 256;
    } 

    // averaging kernel accross all channels
    for (int i = 0; i < kernel_size * kernel_size * kernel_size; i++) {
        h_kernel[i] = 1.0f / (kernel_size * kernel_size * kernel_size);
    }

    float* d_image;
    float* d_kernel;
    float* d_output;
    
    cudaMalloc(&d_image, image_width * image_height * num_channels * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_output, image_width * image_height * num_channels * sizeof(float));
    cudaMemcpy(d_image, h_image, image_width * image_height * num_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);


    // note: can also have a (16, 16, 1) block with a (?, ?, 3) grid (results in more number of blocks and can increase the kernel launch overhead).
    dim3 block_dim(16, 16, 3);
    dim3 grid_dim((image_width + block_dim.x - 1) / block_dim.x, (image_height + block_dim.y - 1) / block_dim.y, 1);

    convolution3d<<<grid_dim, block_dim>>> (d_image, d_kernel, d_output, image_width, image_height, num_channels, kernel_size);

    cudaMemcpy(h_output, d_output, image_width * image_height * num_channels * sizeof(float), cudaMemcpyDeviceToHost);
    
    delete[] h_image;
    delete[] h_kernel;
    delete[] h_output;

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);

}