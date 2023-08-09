#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

template<typename T>
struct MaskAndScaleAddFunctor {
  MaskAndScaleAddFunctor(const uint8_t* mask, const T* addend, float scale)
      : mask(mask), addend(addend), scale(scale) {}
  __device__ T Compute(T x, int64_t i) const {
    return x * static_cast<T>(static_cast<bool>(mask[i]) * scale) + addend[i];
  }
  const uint8_t* mask;
  const T* addend;
  float scale;
};

template<typename FUNCTOR, typename T>
__global__ void FusedBiasAddGpufloat(FUNCTOR functor, const int elem_cnt, const int bias_size,
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
    float* addend = new float[1000];
    for (int i = 0; i < 1000; i++){
        mask_tensor[i] = (uint8_t)(i);
        addend[i] = (float)(i);
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
    MaskAndScaleAddFunctor<float> mask_and_scale_add_functor(mask_tensor, addend, scale);
    FusedBiasAddGpufloat<<<gridSize , blockSize>>>(mask_and_scale_add_functor, ele_cnt, bias_size, d_x, d_bias, d_y);
    cudaMemcpy(y, d_y, sizeof(float) * ele_cnt, cudaMemcpyDeviceToHost);
    printf("pass");
    free(x);
    free(y);
    free(bias);
    delete addend;
    addend = nullptr;
    delete mask_tensor;
    addend = nullptr;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_bias);
}