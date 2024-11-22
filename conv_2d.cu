





#include <iostream>
#include <cuda_runtime.h>
#include <torch/torch.h>
using namespace std;

__global__ void convolution2D(float* input, float* kernel, float* output, int imageWidth, int imageHeight, int kernelSize) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < imageWidth && y < imageHeight) {
        float result = 0.0f;

        for (int i=-kernelSize/2; i<=kernelSize/2; i++) {
            for (int j=-kernelSize/2; j<=kernelSize/2; j++) {
                int xi = x + i;
                int yj = y + j;
                if (xi >= 0 && xi < imageWidth && yj >= 0 && yj < imageHeight) {
                    result += kernel[(i+kernelSize/2) * kernelSize + j + kernelSize/2] * input[xi * imageWidth + yj];
                }
                
            }
        }

        output[x * imageWidth + y] = result;
    }
}


int main() {
    
    // host code
    const int imageWidth = 1024;
    const int imageHeight = 1024;
    const int kernelSize = 3;
    
    float* h_input = new float[imageWidth * imageHeight];
    float* h_kernel = new float[kernelSize * kernelSize];
    float* h_output = new float[imageWidth * imageHeight];

    for (int i = 0; i < imageWidth * imageHeight; i++) {
        h_input[i] = rand() % 256;
    }

    for (int i = 0; i < kernelSize * kernelSize; i++) {
        h_kernel[i] = 1.0f / (kernelSize * kernelSize);
    }

    float* d_input;
    float* d_kernel;
    float* d_output;
    cudaMalloc(&d_input, imageWidth*imageHeight*sizeof(float));
    cudaMalloc(&d_kernel, kernelSize*kernelSize*sizeof(float));
    cudaMalloc(&d_output, imageWidth*imageHeight*sizeof(float));
    cudaMemcpy(d_input, h_input, imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize*kernelSize*sizeof(float), cudaMemcpyHostToDevice);    


    dim3 blockDim(16, 16);
    dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x, (imageHeight + blockDim.y - 1) / blockDim.y);
    
    convolution2D<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, imageWidth, imageHeight, kernelSize);

    cudaMemcpy(h_output, d_output, imageWidth*imageHeight*sizeof(float), cudaMemcpyDeviceToHost);    

    // PyTorch verification
    torch::Tensor input_tensor = torch::from_blob(h_input, {1, 1, imageHeight, imageWidth});
    torch::Tensor kernel_tensor = torch::from_blob(h_kernel, {1, 1, kernelSize, kernelSize});
    torch::Tensor output_tensor = torch::conv2d(input_tensor, kernel_tensor, {}, 1, kernelSize / 2);
    float* output_data = output_tensor.data_ptr<float>();

    bool is_correct = true;
    for (int i = 0; i < imageWidth * imageHeight; i++) {
        if (abs(h_output[i] - output_data[i]) > 1e-5) {
            is_correct = false;
            break;
        }
    }

    if (is_correct) {
        cout << "CUDA convolution is correct!" << endl;
    } else {
        cout << "CUDA convolution is incorrect!" << endl;
    }
    
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}













