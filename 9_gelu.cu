#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
  __host__ __device__ inline const T& operator[](int i) const { return val[i]; }
  __host__ __device__ inline T& operator[](int i) { return val[i]; }
};

__device__ float TanhApprox(float x) {
  //float r;
  //asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  //return r;
  return tanhf(x);
}

template<typename T>
struct GeluFunctor {
  static constexpr T alpha = static_cast<T>(0.7978845608028654);
  static constexpr T beta = static_cast<T>(0.044714998453855515);

  __device__ GeluFunctor() {};

  __device__ T operator()(T x) const {
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T tanh_in = alpha * (x + beta * x * x * x);
    return half * x * (one + tanh(tanh_in));
  }
};

template<>
struct GeluFunctor<half> {
  static constexpr float alpha = GeluFunctor<float>::alpha;
  static constexpr float beta = GeluFunctor<float>::beta;
  GeluFunctor<float> float_functor;

  __device__ GeluFunctor() {};

  __device__ half operator()(const half x) const {
    // Note: when you have ampere GPU, you can enable the line45-50 method to get performance improvement by half intrinsic instead of static_cast half to fp32.
    //const float tanh_in =
    //    __half2float(__float2half_rn(alpha) * (x + __float2half_rn(beta) * x * x * x));
    //const float tanh_out = TanhApprox(tanh_in);
    //return __float2half_rn(0.5f) * x * (__float2half_rn(1.0f) + __float2half_rn(tanh_out));
    // Note: half to float will lose performance using static_cast, because static_cast will be compiled to more instructions than half intrinsic,
    // so you should better use half intrinsic when you have ampere GPU, you can enable 44-47 line
    return static_cast<half>(float_functor(static_cast<float>(x)));
  }
  // Note: when you have ampere GPU, you can enable the "apply2" method to get performance improvement by half2 intrinsic.
  //__device__ void Apply2(half* y, const half* x) const {
    //const half2 x2 = *(reinterpret_cast<const half2*>(x));
    //const float2 tanh_in = __half22float2(
     //   __hmul2(__float2half2_rn(alpha),
      //          __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), x2), x2), x2))));
    //float2 tanh_out;
    //tanh_out.x = TanhApprox(tanh_in.x);
    //tanh_out.y = TanhApprox(tanh_in.y);
    //const half2 y2 = __hmul2(__hmul2(__hmul2(__float2half2_rn(0.5F), x2),
    //                                 __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out))));
    //*reinterpret_cast<half2*>(y) = y2;
  //}
};


template <int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x,
                                                 __half* y,
                                                 int n) {
  int offset =
      static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  int stride = static_cast<int>(blockDim.x * gridDim.x) * VecSize;
  GeluFunctor<half> gelu_fwd;
  __half y_reg[VecSize];
  for (; offset < n; offset += stride) {
    using ArrT = AlignedVector<__half, VecSize>;
    const ArrT* in_arr = reinterpret_cast<const ArrT*>(x + offset);
    // ArrT* out_arr = reinterpret_cast<const ArrT*>(y + offset);
    const __half* in = reinterpret_cast<const __half*>(in_arr);
    // __half* out = reinterpret_cast<const __half*>(out_arr);

    if (VecSize == 1){
        y_reg[0] = gelu_fwd(in[0]);
    } else {
      // Note: when you have ampere GPU, you can enable the "apply2" method to get performance improvement by half2 intrinsic.
      //for (int i = 0; i < VecSize; i+=2) {
      //gelu_fwd.apply2(y + offset, in[i]);
        for (int i = 0; i < VecSize; i++) {
            y_reg[i] = gelu_fwd(in[i]);
        }
    }
    *reinterpret_cast<ArrT*>(y + offset) = *reinterpret_cast<ArrT*>(y_reg);
  }
}

int main() {
    int n = 1000;
    
    __half *x = new __half[n];
    __half *y = new __half[n];
    for (int i = 0; i < n; i++)
    {
      x[i] = (__half)(i);
    }
    __half * d_x, *d_y;
    cudaMalloc((void **)&d_x, n * sizeof(__half));
    cudaMalloc((void **)&d_y, n * sizeof(__half));
    cudaMemcpy(d_x, x, sizeof(__half) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(__half) * n, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    auto is_aligned = [](const void* p, int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };
                                                                      
    constexpr auto kAlignment = alignof(AlignedVector<__half, 8>); 
    // Note: when you have ampere GPU, you can enable the 122-124 line to get performance improvement by half2 intrinsic.
    if (n % 8 == 0 && is_aligned(x, kAlignment) && is_aligned(y, kAlignment)) {                                          
      int thread = std::min<int>(512, deviceProp.maxThreadsPerBlock); 
      //int block = (n / 8 + thread - 1) / thread;                  
      //block = std::min<int>(block, deviceProp.maxGridSize[0]);                                  
      //FP16GeluCUDAKernel<8, true><<<block, thread>>>(x, y, n);  
      int block = (n + thread - 1) / thread;                  
      block = std::min<int>(block, deviceProp.maxGridSize[0]);                                  
      FP16GeluCUDAKernel<1><<<block, thread>>>(d_x, d_y, n);                      
      cudaMemcpy(y, d_y, sizeof(__half) * n, cudaMemcpyDeviceToHost);                                                          
    }   
    printf("pass");
    delete x;
    x = nullptr;
    delete y;
    y = nullptr;
    cudaFree(d_x);
    cudaFree(d_y);
}
