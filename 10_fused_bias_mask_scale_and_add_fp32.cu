#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

template<typename T>
struct MaskScaleAndElementwiseAddFunctor {
  MaskScaleAndElementwiseAddFunctor(const uint8_t* mask, const T* add_val, float scale)
      : mask(mask), add_val(add_val), scale(scale) {}
  __device__ T Compute(T x, int64_t i) const {
    return x * static_cast<T>(static_cast<bool>(mask[i]) * scale) + add_val[i];
  }
  const uint8_t* mask;
  const T* add_val;
  float scale;
};

template<typename FUNCTOR, typename T>
__global__ void FusedBiasAddCUDAKernelFloat(FUNCTOR functor, const int elem_cnt, const int bias_size,
                                const T* x, const T* bias, T* y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elem_cnt;
       i += blockDim.x * gridDim.x){
    T x_i = x[i] + bias[i % bias_size];
    y[i] = functor.Compute(x_i, i);
  }
}

int main(){
    int ele_cnt = 1000;
    float scale = 0.5;
    uint8_t* mask_tensor = new uint8_t[1000];
    float* add_val = new float[1000];
    for (int i = 0; i < 1000; i++){
        mask_tensor[i] = (uint8_t)(i);
        add_val[i] = (float)(i);
    }
    int bias_size = 10;
    float *x = (float*) malloc(sizeof(float) * ele_cnt);
    float *y = (float*) malloc(sizeof(float) * ele_cnt);
    float *bias = (float*) malloc(sizeof(float) * bias_size);
    for (int i = 0; i < ele_cnt; i++)
    {
      x[i] = (float)(i);
    }
    float *d_x, *d_y, *d_bias;
    cudaMalloc((void **)&d_x, ele_cnt * sizeof(float));
    cudaMalloc((void **)&d_y, ele_cnt * sizeof(float));
    cudaMalloc((void **)&d_bias, bias_size * sizeof(float));
    cudaMemcpy(d_x, x, sizeof(float) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * bias_size, cudaMemcpyHostToDevice);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    int blockSize = 256;
    int gridSize = std::min((ele_cnt + blockSize - 1) / blockSize, maxblocks);
    MaskScaleAndElementwiseAddFunctor<float> mask_scale_and_elementwise_add_func(mask_tensor, add_val, scale);
    FusedBiasAddCUDAKernelFloat<<<gridSize , blockSize>>>(mask_scale_and_elementwise_add_func, ele_cnt, bias_size, d_x, d_bias, d_y);
    cudaMemcpy(y, d_y, sizeof(float) * ele_cnt, cudaMemcpyDeviceToHost);
    printf("pass");
    free(x);
    free(y);
    free(bias);
    delete add_val;
    add_val = nullptr;
    delete mask_tensor;
    add_val = nullptr;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_bias);
}
