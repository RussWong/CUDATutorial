#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
typedef __half half;
typedef __half2 half2;

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

template<>
struct MaskAndScaleAddFunctor<half> {
  MaskAndScaleAddFunctor(const uint8_t* mask, const half* addend, float scale)
      : mask(mask), addend(addend), scale(scale) {}
  __device__ half Compute(half x, int64_t i) const {
    return x * static_cast<half>(static_cast<bool>(mask[i]) * scale) + addend[i];
  }
  __device__ half2 ComputeHalf2(half2 x, int64_t i) const {
    const char2* mask_c2 = reinterpret_cast<const char2*>(mask);
    const half2* addend_h2 = reinterpret_cast<const half2*>(addend);
    char2 mask_val = mask_c2[i];
    half2 one_or_zero_h2;
    half2 h2_scale = __float2half2_rn(scale);
    one_or_zero_h2.x = mask_val.x;
    one_or_zero_h2.y = mask_val.y;
    return __hadd2(__hmul2(__hmul2(x, one_or_zero_h2), h2_scale), addend_h2[i]);
  }
  const uint8_t* mask;
  const half* addend;
  float scale;
};


template<typename FUNCTOR>
__global__ void FusedBiasAddGpuHalf2(FUNCTOR functor, const int elem_cnt,
                                        const int bias_size, const half* x, const half* bias,
                                        half* y) {
  const int h2_elem_cnt = elem_cnt / 2;
  const int h2_bias_size = bias_size / 2;
  const auto* x_h2 = reinterpret_cast<const half2*>(x);
  const auto* bias_h2 = reinterpret_cast<const half2*>(bias);
  auto* y_h2 = reinterpret_cast<half2*>(y);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < h2_elem_cnt;
       i += step){
    half2 x_i = __hadd2(x_h2[i], bias_h2[i % h2_bias_size]);
    y_h2[i] = functor.ComputeHalf2(x_i, i);
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
 
    __half *x = (__half*) malloc(sizeof(__half) * ele_cnt);
    __half *y = (__half*) malloc(sizeof(__half) * ele_cnt);
    __half *bias = (__half*) malloc(sizeof(__half) * bias_size);
    for (int i = 0; i < ele_cnt; i++)
    {
      x[i] = (__half)(i);
    }
    __half *d_x, *d_y, *bias;
    cudaMalloc((void **)&d_x, ele_cnt * sizeof(__half));
    cudaMalloc((void **)&d_y, ele_cnt * sizeof(__half));
    cudaMalloc((void **)&d_bias, bias_size * sizeof(__half));
    cudaMemcpy(d_x, x, sizeof(__half) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(__half) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(__half) * bias_size, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    int blockSize = 256;
    int gridSize = std::min((ele_cnt + blockSize - 1) / blockSize, maxblocks);
    MaskAndScaleAddFunctor<half> mask_and_scale_add_functor(mask_tensor, addend, scale);
    FusedBiasAddGpuHalf2<<<gridSize ,blockSize>>>(mask_and_scale_add_functor, ele_cnt, bias_size, d_x, d_bias, d_y);
    cudaMemcpy(y, d_y, sizeof(__half) * ele_cnt, cudaMemcpyDeviceToHost);
    
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