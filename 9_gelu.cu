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
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return r;
#else
  return tanhf(x);
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
}

template<typename T>
struct FusedFastGeluFunctor {
  static constexpr T alpha = static_cast<T>(0.7978845608028654);
  static constexpr T beta = static_cast<T>(0.044714998453855515);

  __device__ FusedFastGeluFunctor() {};

  __device__ T operator()(T x, T m) const {
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T tanh_in = alpha * (x + beta * x * x * x);
    return half * x * (one + tanh(tanh_in)) * m;
  }
};

template<>
struct FusedFastGeluFunctor<half> {
  static constexpr float alpha = FusedFastGeluFunctor<float>::alpha;
  static constexpr float beta = FusedFastGeluFunctor<float>::beta;
  FusedFastGeluFunctor<float> float_functor;

  __device__ FusedFastGeluFunctor() {};

  __device__ half operator()(const half x) const {
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
    const float tanh_in =
        __half2float(__float2half_rn(alpha) * (x + __float2half_rn(beta) * x * x * x));
    const float tanh_out = TanhApprox(tanh_in);
    return __float2half_rn(0.5F) * x * (__float2half_rn(1.0F) + __float2half_rn(tanh_out));
#else
    return static_cast<half>(float_functor(static_cast<float>(x)));
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  }

//#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  __device__ void Apply2(half* y, const half* x) const {
    const half2 x2 = *(reinterpret_cast<const half2*>(x));
    const float2 tanh_in = __half22float2(
        __hmul2(__float2half2_rn(alpha),
                __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), x2), x2), x2))));
    float2 tanh_out;
    tanh_out.x = TanhApprox(tanh_in.x);
    tanh_out.y = TanhApprox(tanh_in.y);
    const half2 y2 = __hmul2(__hmul2(__hmul2(__float2half2_rn(0.5F), x2),
                                     __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out))));
    *reinterpret_cast<half2*>(y) = y2;
  }
//#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
};


template <int VecSize, bool FastMode>
__global__ void FP16FastGeluFwdCUDAKernel(const __half* x,
                                                 __half* y,
                                                 size_t n) {
  size_t offset =
      static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * VecSize;
  FusedFastGeluFunctor<half> gelu_fwd;
  for (; offset < n; offset += stride) {
    using ArrT = AlignedVector<__half, VecSize>;
    ArrT* in_arr = reinterpret_cast<const ArrT*>(x + offset);
    // ArrT* out_arr = reinterpret_cast<const ArrT*>(y + offset);
    __half* in = reinterpret_cast<const __half*>(in_arr);
    // __half* out = reinterpret_cast<const __half*>(out_arr);
#pragma unroll
    for (int i = 0; i < VecSize; i+=2) {
      // gelu_fwd.apply2(out[i], in[i]);
      gelu_fwd.apply2(y + offset, in[i]);
    }
    // reinterpret_cast<const ArrT*>(y + offset) = *reinterpret_cast<const ArrT*>(out);
  }
}

int main() {
// static bool TryLaunchFP16FastGeluFwdVectorizeCUDAKernel(
//     const GPUContext& dev_ctx, const __half* x, __half* y, size_t n) {
    int n = 1000;
    
    __half *x = new __half[n];
    __half *y = new __half[n];
    for (size_t i = 0; i < n; i++)
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

    auto is_aligned = [](const void* p, size_t alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

// #define PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL(__vec_size, __use_fast_math)      \
//   do {                                                                        
    constexpr auto kAlignment = alignof(AlignedVector<__half, 8>);                      
    if (n % 8 == 0 && is_aligned(x, kAlignment) && is_aligned(y, kAlignment)) {                                          
      size_t thread = std::min<size_t>(512, deviceProp.GetMaxThreadsPerBlock()); 
      size_t block = (n / 8 + thread - 1) / thread;                  
      block = std::min<size_t>(block, deviceProp.GetCUDAMaxGridDimSize()[0]);                                  
      FP16FastGeluFwdCUDAKernel<8, true><<<block, thread>>>(x, y, n);                  
      cudaMemcpy(y, d_y, sizeof(__half) * n, cudaMemcpyDeviceToHost);                                                          
    }   

    delete x;
    x = nullptr;
    delete y;
    y = nullptr;
    cudaFree(d_x);
    cudaFree(d_y);
                                                                   
//   } while (0)

//   if (FLAGS_use_fast_math) {
//     PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL(8, true);
//   } else {
//     PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL(8, false);
//   }

// #undef PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL
}
