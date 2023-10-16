#include <cmath>
#include <random>
#include <bits/stdc++.h>
#include <float.h>
#include <cuda.h>
#include "cuda_runtime.h"
// still have bug, waiting for correct..
bool CheckResult(float *out, float* groudtruth, int nums){
    for (int i = 0; i < nums; i++){
      if (groudtruth[i] != out[i]) {
        return false;
      }
    }
    return true;
}
// python version
// def gen_quant_scale_for_min_max_symmetric(weight, quantization_bit):
//     weight_max = np.max(np.abs(weight))
//     denominator = 2.0 ** (quantization_bit - 1) - 1
//     return (weight_max / denominator, 0)

template<typename T>
void GenScalePerTensorSymmetricCPU(const T* in_ptr, const int quantization_bit,
                            const int num_elements, T* scale, T* zero_point) {
  T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
  T in_min = *std::min_element(in_ptr, in_ptr + num_elements);
  in_max = std::max(std::abs(in_max), std::abs(in_min));
  T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  *scale = in_max / denominator;
  *zero_point = 0;
}

// python version
// def gen_quant_scale_for_min_max_affine(weight, quantization_bit):
//     weight_max = np.max(weight)
//     weight_min = np.min(weight)
//     denominator = 2.0 ** quantization_bit - 1
//     scale = (weight_max - weight_min) / denominator
//     zero_point = -np.round(weight_min / scale)
//     return (scale, zero_point)

template<typename T>
void QuantizationPerTensorSymmetricCPU(const T* in_ptr, const T scale, const int quantization_bit,
                                   const int num_elements, T* out_ptr) {
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -1 * upper_bound - 1;
  for(int j = 0; j < num_elements; j++) {
    T out = std::nearbyint(in_ptr[j] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[j] = out;
  }
}

// def quant_per_layer_affine(input, quantization_bit, scale, zero_point):
//     upper_bound = 2.0 ** quantization_bit - 1
//     lower_bound = 0
//     return np.clip(np.rint(input / scale + zero_point), lower_bound, upper_bound)

inline __device__ float atomicMax(float *addr, float value) {
  float old = *addr, assumed;
  if (old >= value) return old;
  do {
    assumed = old;
    old = atomicCAS((unsigned int *)addr, __float_as_int(assumed),
                    __float_as_int(value));

  } while (old != assumed);

  return old;
}

inline __device__ float atomicMin(float *addr, float value) {
  float old = *addr, assumed;
  if (old <= value) return old;
  do {
    assumed = old;
    old = atomicCAS((unsigned int *)addr, __float_as_int(assumed),
                    __float_as_int(value));

  } while (old != assumed);

  return old;
}

// from min_max_observer_kernel.cu
// get max and min per tensor
// use block shared memory reduce
template<typename T>
__global__ void ReduceMaxMinPerTensor(const T* input_ptr, const int nums, T* max_ptr,
                                     T* min_ptr) {
  // dyn shared memory
  extern __shared__ unsigned char shared_max_min_memory[];
  T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
  T* shared_min = shared_max + blockDim.x;
  int total_thread_num = blockDim.x * gridDim.x;
  // follow the reduce v4
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + tid;
  shared_max[tid] = FLT_MAX;
  shared_min[tid] = FLT_MIN;

  for (int i = gid; i < nums; i += total_thread_num) {
      shared_max[tid] = max(shared_max[tid], input_ptr[gid]);
      shared_min[tid] = min(shared_min[tid], input_ptr[gid]);
  }
  __syncthreads();
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && gid < nums) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
      shared_min[tid] = max(shared_min[tid], shared_min[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
      atomicMax(max_ptr, shared_max[0]);
      atomicMin(min_ptr, shared_min[0]);
  }
}

// get max and min per channel
template<typename T>
__global__ void ReduceMaxMinPerChannel(const T* input_ptr, const int nums,
                                       const int num_channels, const int HW,
                                       T* max_ptr, T* min_ptr) {
  extern __shared__ unsigned char shared_max_min_memory[];
  T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
  T* shared_min = shared_max + blockDim.x;

  int cur_channel = blockIdx.x;
  int tid = threadIdx.x;

  while (cur_channel < num_channels) {
    shared_max[tid] = FLT_MAX;
    shared_min[tid] = FLT_MIN;

    int index = (HW * cur_channel) + tid;
    int end = HW * (cur_channel + 1);

    while (index < end && index < nums) {
      shared_max[tid] = max(shared_max[tid], input_ptr[index]);
      shared_min[tid] = min(shared_min[tid], input_ptr[index]);
      index += blockDim.x;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      atomicMax(&max_ptr[cur_channel], shared_max[0]);
      atomicMin(&min_ptr[cur_channel], shared_min[0]);
    }
    cur_channel += gridDim.x;
  }
}

template<typename T>
__global__ void GetScaleAndZPSymmetric(const T* max_ptr, const T* min_ptr,
                                           const int nums, const double quantization_bit,
                                           T* scale, T* zero_point) {
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + tid;
  while (gid < nums) {
    T weight_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    scale[gid] = weight_max / denominator;
    zero_point[gid] = 0;
    gid += gridDim.x * blockDim.x;
  }
}
// HW = h * w
template<typename T>
__global__ void GetScaleAndZPAsymmetric(const T* max_ptr, const T* min_ptr, const int nums,
                                        const double quantization_bit, T* scale, T* zero_point) {
  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid;
  while (gid < nums) {
    T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;
    T min = -min_ptr[gid];
    T s = (max_ptr[gid] - min) / denominator;
    scale[gid] = s;
    zero_point[gid] = -1 * std::nearbyint(min / s);
    gid += gridDim.x * blockDim.x;
  }
}

template<typename T>
__global__ void QuantizePerChannelSymmetric(const T* in_ptr, const T* scale_ptr, 
                                  const int scale_size, const int nums, const int HW,
                                  const double quantization_bit, T* out_ptr) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;

  while (gid < nums) {
    int channel_index = gid / HW;
    int scale_idx = min(scale_size - 1, channel_index);
    T scale = scale_ptr[scale_idx];

    T out = std::nearbyint(in_ptr[gid] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out;

    gid += step;
  }
}

template<typename T>
__global__ void QuantizePerChannelAsymmetric(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                                   const int scale_size, const int nums,
                                   const int HW, const double quantization_bit,
                                   T* out_ptr) {
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit)) - 1;
  T lower_bound = 0;

  while (gid < nums) {
    int channel_index = gid / HW;
    int scale_idx = min(scale_size - 1, channel_index);

    T scale = scale_ptr[scale_idx];
    T zero_point = zero_point_ptr[scale_idx];

    T out = std::nearbyint(in_ptr[gid] / scale + zero_point);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out;

    gid += step;
  }
}

template<typename T>
__global__ void QuantizePerTensorSymmetric(const T* in_ptr, const T* scale_ptr, 
                                  const int nums, const int HW, const double quantization_bit, T* out_ptr) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;

  while (gid < nums) {
    T scale = *scale_ptr;

    T out = std::nearbyint(in_ptr[gid] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out;

    gid += step;
  }
}

template<typename T>
__global__ void QuantizePerTensorAsymmetric(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                                   const int nums, const int HW, const double quantization_bit, T* out_ptr) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit)) - 1;
  T lower_bound = 0;

  while (gid < nums) {
    T scale = *scale_ptr;
    T zero_point = *zero_point_ptr;

    T out = nearbyint(in_ptr[gid] / scale + zero_point);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out;

    gid += step;
  }
}


int main() {
  float milliseconds = 0;
  constexpr int nums = 400 * 20 * 10;
  constexpr int HW = 20 * 10;
  constexpr int channel = 400;
  constexpr int quantization_bit = 8;
  float* input = (float*) malloc(sizeof(float) * nums);
  for(int i = 0; i < nums; i++) {
    // generate float input inside [-1, 1]
    input[i] = -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/2));
  }
  float* output = (float*) malloc(sizeof(float) * nums);
  float *d_input, *d_output;
  cudaMalloc((void **)&d_input, nums * sizeof(float));
  cudaMalloc((void **)&d_output, nums * sizeof(float));
  cudaMemcpy(d_input, input, sizeof(float) * nums, cudaMemcpyHostToDevice);
  // block and thread config
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int maxblocks = deviceProp.maxGridSize[0];
  int blockSize = 256;
  int gridSize = std::min<int>((nums + blockSize - 1) / blockSize,  std::min<int>(maxblocks, channel));

  float *d_scale, *d_zeropoint, *d_max, *d_min;
  // per tensor, scale and zp shape both are 1
  // switch to per tensor
  bool per_tensor_quantize = true; 
  if(per_tensor_quantize) {
    cudaMalloc((void **)&d_scale, 1 * sizeof(float));
    cudaMalloc((void **)&d_zeropoint, 1 * sizeof(float)); 
    cudaMalloc((void **)&d_max, 1 * sizeof(float));
    cudaMalloc((void **)&d_min, 1 * sizeof(float));  
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReduceMaxMinPerTensor<float><<<gridSize, blockSize>>>(d_input, nums, d_max, d_min);
    GetScaleAndZPSymmetric<float><<<gridSize, blockSize>>>(d_max, d_min, nums, quantization_bit, d_scale, d_zeropoint);
    QuantizePerTensorSymmetric<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, HW, quantization_bit, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
  } else {
  // perchannel, shape = channel
    cudaMalloc((void **)&d_scale, channel * sizeof(float));
    cudaMalloc((void **)&d_zeropoint, channel * sizeof(float)); 
    cudaMalloc((void **)&d_max, channel * sizeof(float));
    cudaMalloc((void **)&d_min, channel * sizeof(float)); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReduceMaxMinPerChannel<float><<<gridSize, blockSize>>>(d_input, nums, channel, HW, d_max, d_min);
    GetScaleAndZPSymmetric<float><<<gridSize, blockSize>>>(d_max, d_min, nums, quantization_bit, d_scale, d_zeropoint);
    QuantizePerChannelSymmetric<float><<<gridSize, blockSize>>>(d_input, d_scale, channel, nums, HW, quantization_bit, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
  }
    cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost);
  // (per tensor) get CPU output to validate GPU result is right or not
  float* CPUOutput= (float*) malloc(sizeof(float) * nums);
  float* scale = (float*) malloc(sizeof(float) * 1);
  float* zeropoint = (float*) malloc(sizeof(float) * 1);
  GenScalePerTensorSymmetricCPU<float>(input, quantization_bit, nums, scale, zeropoint);
  QuantizationPerTensorSymmetricCPU<float>(input, *scale, quantization_bit, nums, CPUOutput);
  if (CheckResult(output, CPUOutput, nums)) {
    printf("the ans is right");
  } else {
    printf("the ans is wrong\n");
    printf("first two CPUoutput are %f, %f\n", CPUOutput[0], CPUOutput[1]);
    printf("first two output are %f, %f\n", output[0], output[1]);
  }
  printf("Quantize kernel latency = %f ms\n", milliseconds);
  free(input);
  free(output);
  free(CPUOutput);
  free(scale);
  free(zeropoint);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_scale);
  cudaFree(d_zeropoint);
  cudaFree(d_max);
  cudaFree(d_min);
}
