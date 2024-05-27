#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)
// 自定义向量化类型，主要用于VecType_u8
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};
// 随机数生成器
template <typename T>
struct uniform_distribution {
  __device__ T operator()(curandStatePhilox4_32_10_t* state) {
    return static_cast<T>(curand_uniform(state));
  }
  static constexpr int Count = 1;
};

template <>
struct uniform_distribution<float> {
  __device__ float4 operator()(curandStatePhilox4_32_10_t* state) {
    return curand_uniform4(state);
  }
  static constexpr int Count = 4;
};
// 计算输出和mask的最小逻辑
template <typename T>
struct DstMaskFunctor {
  const float prob_;
  const bool is_upscale_in_train_;
  float inv_prob;
  __device__ DstMaskFunctor(const float prob, const bool is_upscale_in_train)
      : prob_(prob), is_upscale_in_train_(is_upscale_in_train) {
    inv_prob = 1.0f / (1 - prob_);
  }

  __device__ void operator()(T* dst, const T* src_val, const T* rand) {
    static constexpr int Count = uniform_distribution<T>::Count;
    for (int i = 0; i < Count; i++) {
      if (rand[i] < prob_) {
        dst[i] = static_cast<T>(0);
        dst[i + Count] = dst[i];
      } else {
        dst[i] = is_upscale_in_train_ ? static_cast<T>(src_val[i] * inv_prob)
                                      : static_cast<T>(src_val[i]);
        dst[i + Count] = static_cast<T>(1);
      }
    }
  }
};

template <typename T, typename MaskType>
__global__ void VectorizedDstMask(const size_t n,
                       int seed,
                       const float dropout_prob,
                       const T* src,
                       MaskType* mask,
                       T* dst,
                       bool is_upscale_in_train,
                       int increment,
                       int main_offset) {
  int thread_idx = threadIdx.x;
  int thread_nums = blockDim.x;
  int block_idx = blockIdx.x;
  int block_nums = gridDim.x;
  
  int block_offset = block_idx * thread_nums;
  constexpr int VecSize = uniform_distribution<float>::Count;
  // 所有block在一次迭代中的数据处理总量
  int stride = block_nums * thread_nums * VecSize;
  // 初始化随机数状态
  curandStatePhilox4_32_10_t state;
  curand_init(seed, block_offset + thread_idx, increment, &state);
  // 声明相关寄存器，暂存输出数据、随机数、mask
  T dst_mask[VecSize * 2];  // 0 ~ VecSize -1 : dst;VecSize ~ 2 * VecSize - 1: mask
  float rands[VecSize];
  MaskType mask_result[VecSize];
  // 初始化生成随机数的functor、计算mask和输出结果的functor
  using Rand = uniform_distribution<float>;
  auto dst_functor =
      DstMaskFunctor<T>(dropout_prob, is_upscale_in_train);

  using VecType = float4;
  using VecType_u8 = VectorType<MaskType, VecSize>;
  VecType vec_temp_input;
  // 可以向量化的部分
  int start = block_offset * VecSize;
  for (; start < main_offset; start += stride) {
    // 取出数据
    int thread_offset = thread_idx;
    const VecType* vec_input = reinterpret_cast<const VecType*>(src + start);
    vec_temp_input = vec_input[thread_offset];
    auto random_tuple = Rand()(&state);
    for (int i = 0; i < VecSize; i++) {
      dst_mask[i] = *(reinterpret_cast<T*>(&vec_temp_input) + i);
      rands[i] = static_cast<float>((&random_tuple.x)[i]);
    }
    // 算出数据
    dst_functor(&dst_mask[0], &dst_mask[0], &rands[0]);
    // 写回数据
    T* res = dst + start;
    VecType* vec_dst_output = reinterpret_cast<VecType*>(res);
    vec_dst_output[thread_offset] = *(reinterpret_cast<VecType*>(&dst_mask[0]));

    for (int i = 0; i < VecSize; i++) {
      mask_result[i] = static_cast<MaskType>(dst_mask[i + VecSize]);
    }

    MaskType* mask_res = mask + start;
    VecType_u8* vec_mask_output = reinterpret_cast<VecType_u8*>(mask_res);
    vec_mask_output[thread_offset] =
        *(reinterpret_cast<VecType_u8*>(mask_result));
  }
  // 不可向量化的部分
  int remain = n - start;
  if (remain > 0) {
    // 取出数据
    int thread_offset = thread_idx * VecSize;
    const T* src_remain = src + start;
    auto random_tuple = Rand()(&state);
    for (int i = 0; i < VecSize; i++) {
      if (i + thread_offset < remain) {
        dst_mask[i] = src_remain[thread_offset + i];
      }
      rands[i] = static_cast<float>((&random_tuple.x)[i]);
    }
    // 算出数据
    dst_functor(&dst_mask[0], &dst_mask[0], &rands[0]);
    // 写回数据
    T* res = dst + start;
    MaskType* mask_res = mask + start;
    for (int i = 0; i < VecSize; i++) {
      if ((thread_offset + i) < remain) {
        res[thread_offset + i] = dst_mask[i];
        mask_result[i] = static_cast<MaskType>(dst_mask[i + VecSize]);
        mask_res[thread_offset + i] = mask_result[i];
      }
    }
  }
}

template <typename T>
void DropoutKernel(const bool is_test,
                  const bool is_upscale_in_train,
                  const size_t num_eles,
                  const float dropout_prob,
                  const int seed_val,
                  const float* x_data,
                  uint8_t* mask_data,
                  float* y_data) {
  // 1. 训练: dropout最多用的场景，丢弃某些神经元
  if (!is_test) {
    if (dropout_prob == 1.0f) {
      cudaMemset(y_data, 0, num_eles);
      cudaMemset(mask_data, 0, num_eles);
      return;
    }

    // 每个线程负责生成4个随机数
    constexpr int RandVecSize = uniform_distribution<float>::Count;

    size_t num_blocks = 2;
    size_t block_size = 256;
    dim3 grid(num_blocks);
    dim3 block(block_size);

    int seed_data = seed_val;
    int increment = 0;
    // 可向量化读写的数据量
    int main_offset =
        num_eles / (num_blocks * block_size * RandVecSize) * (num_blocks * block_size * RandVecSize);

    VectorizedDstMask<T, uint8_t><<<grid, block>>>(num_eles,
                                                  seed_data,
                                                  dropout_prob,
                                                  x_data,
                                                  mask_data,
                                                  y_data,
                                                  is_upscale_in_train,
                                                  increment,
                                                  main_offset);
  } else {
    // 2. 推理场景，output=input
    cudaMemcpy(y_data, x_data, num_eles, cudaMemcpyDeviceToDevice);
  }
}

int main() {
    constexpr size_t num_eles = 2050;// 512 * 4 + 2
    
    float* x = (float*)malloc(num_eles * sizeof(float));
    float* d_x;
    CHECK(cudaMalloc((void **)&d_x, num_eles * sizeof(float)));

    float* y = (float*)malloc(num_eles * sizeof(float));
    float* d_y;
    CHECK(cudaMalloc((void **)&d_y, num_eles * sizeof(float)));

    uint8_t* mask = (uint8_t*)malloc(num_eles * sizeof(uint8_t));
    uint8_t* d_mask;
    CHECK(cudaMalloc((void **)&d_mask, num_eles * sizeof(uint8_t)));

    for(int i = 0; i < num_eles; i++){
        x[i] = 1;
    }

    CHECK(cudaMemcpy(d_x, x, num_eles * sizeof(float), cudaMemcpyHostToDevice));
    const bool is_test = false;
    const bool is_upscale_in_train = true;
    const float dropout_prob = 0.5;
    const int seed_val = 10000;    
    DropoutKernel<float>(is_test,
                        is_upscale_in_train,
                        num_eles,
                        dropout_prob,
                        seed_val,
                        d_x,
                        d_mask,
                        d_y);
    CHECK(cudaMemcpy(y, d_y, num_eles * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(mask, d_mask, num_eles * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    // 打印最后位于可向量化和不可向量化边界的三个结果
    for (int i = num_eles - 3; i < num_eles; i++){
      printf("[%d] y is %f\n",i, y[i]);
      printf("[%d] mask is %d\n",i, mask[i]);
    }
    cudaFree(d_x);                                                                                                        
    cudaFree(d_y);                                                                                                        
    cudaFree(d_mask);                                                                                                        
    free(x);                                                                                                              
    free(y);                                                                                                              
    free(mask);
}