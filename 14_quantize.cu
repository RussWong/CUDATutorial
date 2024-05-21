#include <cmath>
#include <cfenv>
#include <random>
#include <bits/stdc++.h>
#include <float.h>
#include <cuda.h>
#include "cuda_runtime.h"
// Kernel performance:
// PerTensor + Sym: 0.58ms
// PerChannel + Sym: 0.46ms
// solved bugs:
// 1. gpu_res[0] = 0, cpu_res[0] = 86
//     cpu_max is right, gpu_max = very big
// 2. cpu_min != gpu_min, cpu_max != gpu_max, check minmax kernel and guess it resulted from cuda kernel某些地方写错 or atomicMax
// 3. cpu_scale != gpu_scale, ep. cpu_scale = 0.22035, gpu_scale = 0.22036
// 4. cpu_res != gpu_res , ep. cpu_res = 44, gpu_res = 45
// debug tips:
// 1. input some simple case to confirm cpu impl is right
// 1. printf cpu res and gpu res of each kernel
// 2. use if(tid==0) to get the gpu output and key variable of one thread
// 3. use grid step loop to conveniently debug by launch one thread
// 注意: 正在添加图解
bool CheckResult(float *out, float* groudtruth, int nums){
    for (int i = 0; i < nums; i++){
      if (groudtruth[i] != out[i]) {
        printf("the wrong index is %d, the groudtruth is %f, the res is %f\n", i, groudtruth[i], out[i]);
        return false;
      }
    }
    return true;
}
// CPU version equation
// PerTensor + Sym: scale = max(abs(weight)) / 127 , zeropoint = 0, input_int8 = clamp(input_fp32/scale ,-128, 127)
// PerTensor + Asym: scale = (max(weight) - min(weight)) / 255, zeropoint = -round(min(weight))/scale
// PerChannel + Sym: scale[channel_id] = max(abs(weight[channel_id])) / 127 , zeropoint = 0, input_int8[channel_id * HW + (channel_id + 1) * HW] = clamp(input_fp32[channel_id * HW + (channel_id + 1) * HW]/scale[channel_id] ,-128, 127)
// PerChannel + Asym: scale[channel_id] = (max(weight[channel_id]) - min(weight[channel_id])) / 255, zeropoint[channel_id] = -round(min(weight[channel_id]))/scale[channel_id]

// py code
// def gen_quant_scale_for_min_max_symmetric(weight, quantization_bit):
//     weight_max = np.max(np.abs(weight))
//     denominator = 2.0 ** (quantization_bit - 1) - 1
//     return (weight_max / denominator, 0)

template<typename T>
void GenScalePerTensorSymmetricCPU(const T* in_ptr, const int quantization_bit,
                            const int num_elements, T* scale, T* zero_point) {
  T in_max = *std::max_element(in_ptr, in_ptr + num_elements);// absmax
  T in_min = *std::min_element(in_ptr, in_ptr + num_elements);// absmin
  T out_max = std::max(std::abs(in_max), std::abs(in_min));// Wmax
  // printf("weight_max_cpu is %f, std::abs(in_max) is %f, std::abs(in_min) is %f\n",out_max,std::abs(in_max),std::abs(in_min));
  T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  *scale = out_max / denominator; // Wmax / 127
  *zero_point = 0;
}
// py code
// def gen_quant_scale_for_min_max_affine(weight, quantization_bit):
//     weight_max = np.max(weight)
//     weight_min = np.min(weight)
//     denominator = 2.0 ** quantization_bit - 1
//     scale = (weight_max - weight_min) / denominator
//     zero_point = -np.round(weight_min / scale)
//     return (scale, zero_point)

// 公式: clip(input / scale .round(), -128, 127)
template<typename T>
void QuantizationPerTensorSymmetricCPU(const T* in_ptr, const T scale, const int quantization_bit,
                                   const int num_elements, T* out_ptr) {
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;
  // printf("scaleCPU is %f\n", scale);
  for(int j = 0; j < num_elements; j++) {
    T out = std::nearbyint(in_ptr[j] / scale);
    //if (j==328) printf("in_ptrCPU is %f, outCPU is %f\n", in_ptr[j], out);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[j] = out;
  }
}

// def quant_per_layer_affine(input, quantization_bit, scale, zero_point):
//     upper_bound = 2.0 ** quantization_bit - 1
//     lower_bound = 0
//     return np.clip(np.rint(input / scale + zero_point), lower_bound, upper_bound)

template<typename T>
void GenScalePerChannelSymmetricCPU(const T* in_ptr, const int quantization_bit, const int HW, const int channel,
                            const int num_elements, T* scale, T* zero_point) {
   // 与per tensor唯一不同在于每个channel的scale不同，所以循环求scale
  for (int cid = 0; cid < channel; cid++){
    int start = cid * HW;
    int end = (cid + 1) * HW;
    T channel_max = *std::max_element(in_ptr + start, in_ptr + end); // absmax
    T channel_min = *std::min_element(in_ptr + start, in_ptr + end);// note: cannot use [] which get a float, must use + to get pointer
    T out_max = std::max(std::abs(channel_max), std::abs(channel_min));
    // printf("weight_max_cpu is %f, std::abs(in_max) is %f, std::abs(in_min) is %f\n",out_max,std::abs(in_max),std::abs(in_min));
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    scale[cid] = out_max / denominator;
    zero_point[cid] = 0;
  }
}
template<typename T>
void QuantizationPerChannelSymmetricCPU(const T* in_ptr, const T* scale, const int quantization_bit, const int HW,
                                   const int num_elements, T* out_ptr) {
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;
  //printf("scaleCPU is %f\n", scale);
  for(int j = 0; j < num_elements; j++) {
    // j / HW索引到当前元素的channel ID，然后取出对应channel的scale，做quantize
    T out = std::nearbyint(in_ptr[j] / scale[j / HW]);
    //if (j==328) printf("in_ptrCPU is %f, outCPU is %f\n", in_ptr[j], out);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[j] = out;
  }
}
// 以上是CPU上的quantize函数，接下来用CUDA重写
// GPU device function
__device__ float gpunearbyint(float a) {
  return std::nearbyint(a);
}

//about CAS: https://blog.csdn.net/m0_52153904/article/details/130095643
//int atomicCAS(int* address, int compare, int val)
//{
//    old = *address;
//    if(old == compare)
//        *address = val;
//    else
//        *address = old;
//    return(old);
//}

// Note: another version of float type atomicMax
//inline __device__ float atomicMax(float *addr, float value) {
//  float old = *addr, assumed;
//  if (old >= value) return old;
//  do {
//    assumed = old;
//    if (assumed > value) break;
//    old = atomicCAS((unsigned int *)addr, __float_as_int(assumed),
  //                  __float_as_int(value));

//  } while (old != assumed);

//  return old;
//}
// 封装好的atmoicMax不支持fp32类型，所以我们这里需要针对fp32类型重载atomicMax
// fp32 type atomicMax from stackoverflow and nv developer forum: https://forums.developer.nvidia.com/t/cuda-atomicmax-for-float/194207
inline __device__ float atomicMax(float *address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,  __float_as_int(fmaxf(val, __int_as_float(assumed))));

  } while (old != assumed);

  return __int_as_float(old);
}

inline __device__ float atomicMin(float *address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,  __float_as_int(fminf(val, __int_as_float(assumed))));

  } while (old != assumed);

  return __int_as_float(old);
}

// get max and min per tensor
// use block shared memory reduce，即reduce v4，唯一区别只是由reduce_sum变成了reduce_max_min
template<typename T>
__global__ void ReduceMaxMinPerTensor(const T* input_ptr, const int nums, T* max_ptr,
                                     T* min_ptr, const int channel, const int HW) {
  // dyn shared memory
  extern __shared__ unsigned char shared_max_min_memory[];
  T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
  T* shared_min = shared_max + blockDim.x;
  int total_thread_num = blockDim.x * gridDim.x;
  // follow the reduce v4
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + tid;
  shared_max[tid] = FLT_MIN;
  shared_min[tid] = FLT_MAX;
  // 1. block数量可能无法覆盖总数据量，先以total_thread_num把block和block范围外的数据给比较一遍
  for (int i = gid; i < nums; i += total_thread_num) {
      shared_max[tid] = max(shared_max[tid], input_ptr[i]);
      shared_min[tid] = min(shared_min[tid], input_ptr[i]);
      //if(i <= 3){
        //printf("shared max = %f\n", shared_max[tid]);
        //printf("shared_min = %f\n", shared_min[tid]);
      //}
  }
  __syncthreads();
  // 2. 至此，所有block已经覆盖总数据量，于是开始在block内部先比较大小，又称intra-block范围的比较
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && gid < nums) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
      shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
    }
    __syncthreads();
  }
  // 3. 最后，每个block里面的shared mem的0号位置都保存了block内部的最大最小值，此时使用atomic对所有block来进行比较
  if (tid == 0) {
      atomicMax(max_ptr, shared_max[0]);
      atomicMin(min_ptr, shared_min[0]);
      //printf("max = %f\n", *max_ptr);
      //printf("min = %f\n", *min_ptr);
  }
}

// get max and min per channel
template<typename T>
__global__ void ReduceMaxMinPerChannel(const T* input_ptr, const int nums,
                                       T* max_ptr, T* min_ptr, const int num_channels, const int HW) {
  extern __shared__ unsigned char shared_max_min_memory[];
  // 动态smem需要如下这样去强转成我们的计算类型，以及分配每块的大小，比如L224的+blockDim.x就定义shared_max指向blockDim.x个元素
  T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
  T* shared_min = shared_max + blockDim.x;
  // block id represent channel id, if block nums < channel nums or thread nums < HW, we use a loop on 239 and 259 line. 
  int cur_channel = blockIdx.x;
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  // get min/max of each channel
  while (cur_channel < num_channels) {
    shared_max[tid] = FLT_MIN;
    shared_min[tid] = FLT_MAX;
    // thread offset and end offset of each channel
    // index表示每个线程所取的元素位置，位于第cur channel的第tid偏移
    int index = (HW * cur_channel) + tid;
    int end = HW * (cur_channel + 1);
    // 确定好了index，其他与reduceMinMaxPerTensor差不多
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
      if(blockIdx.x==0){
        printf("max = %f\n", max_ptr[0]);
        printf("min = %f\n", min_ptr[0]);
      }
    }
    cur_channel += gridDim.x;
  }
}

// Note: performance will be low if use cpu function
// why? min max ptr locate GPU memory ,not CPU memory.Because CPU funtion requests CPU memory data, so need
// copy max min ptr from GPU memory to CPU memory, which will incur some overhead!
// in addition, next kernel will run on GPU, at that time, scale and zeropoint will be copied from host to device,which will incur overhead, too
template<typename T>
__global__ void GetScaleAndZPSymmetric(const T* max_ptr, const T* min_ptr,
                                           const int nums, const double quantization_bit,
                                           T* scale, T* zero_point) {
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + tid;
  while (gid < nums) {
    T weight_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));
    //if (gid==0) printf("weight_max_gpu is %f, fabs(max_ptr[gid]) is %f, fabs(min_ptr[gid]) is %f\n",weight_max,fabs(max_ptr[gid]),fabs(min_ptr[gid]));
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
__global__ void QuantizePerChannelSymmetric(const T* in_ptr, const T* scale_ptr, const int nums,
                                  const double quantization_bit, T* out_ptr,
                                  const int scale_size, const int HW) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;
  // 逐个元素做按照quantize公式做quantize，注意channel ID要先取到，然后去取该channel对应的scale
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

// element wise operation
template<typename T>
__global__ void QuantizePerTensorSymmetric(const T* in_ptr, const T* scale_ptr,
                                  const int nums, const double quantization_bit, T* out_ptr, const int channel, const int HW) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;
  T scale = *scale_ptr;
  if (gid==0) printf("scaleGPU is %f\n", scale);
  while (gid < nums) {
    // per tensor quant和per channel quant的最大区别在于这里不用去根据channel ID取对应的scale，而是整个tensor共用一个scale
    T out = gpunearbyint(in_ptr[gid] / scale);
    if (gid==328) printf("328 in_ptr is %f, out is %f\n", in_ptr[gid], out);
    if (gid==1587) printf("1587 in_ptr is %f, out is %f\n", in_ptr[gid], out);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out;

    gid += step;
  }
}

template<typename T>
__global__ void QuantizePerTensorAsymmetric(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                                   const int nums, const double quantization_bit, T* out_ptr,
                                  const int channel, const int HW) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit)) - 1;
  T lower_bound = 0;
  T scale = *scale_ptr;
  T zero_point = *zero_point_ptr;
  while (gid < nums) {

    T out = nearbyint(in_ptr[gid] / scale + zero_point);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out;

    gid += step;
  }
}

// use macro to reduce redundant code
#define LAUNCH_GPU_KERNEL(GetMinMaxFunc, QuantFunc, scale_size, channel, HW) \
    cudaMalloc((void **)&d_scale, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_zeropoint, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_max, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_min, scale_size * sizeof(float)); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    GetMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min, channel, HW);  \
    GetScaleAndZPSymmetric<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale, d_zeropoint); \
    QuantFunc<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output, channel, HW); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&milliseconds, start, stop);

int main() {
  float milliseconds = 0;
  constexpr int nums = 400 * 20 * 10;
  constexpr int HW = 20 * 10;
  constexpr int channel = 400;
  constexpr int quantization_bit = 8;
  float* input = (float*) malloc(sizeof(float) * nums);
  float cpu_min = FLT_MAX;
  float cpu_max = FLT_MIN;
  for(int i = 0; i < nums; i++) {
    // generate float input inside [-1, 1],[-3,3]
    input[i] = -3 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/6));
    cpu_min = std::min(input[i], cpu_min);
    cpu_max = std::max(input[i], cpu_max);
  }
  // printf("per tensor min max cpu are  %f, %f\n", cpu_min, cpu_max);
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
  printf("gridsize blocksize are  %d, %d\n", gridSize, blockSize);
  float *d_scale, *d_zeropoint, *d_max, *d_min;
  // per tensor, scale and zp shape both are 1
  // switch to per tensor
  bool per_tensor_quantize = false;
  if(per_tensor_quantize) {
    //cudaMalloc((void **)&d_scale, 1 * sizeof(float));
    //cudaMalloc((void **)&d_zeropoint, 1 * sizeof(float));
    //cudaMalloc((void **)&d_max, 1 * sizeof(float));
    //cudaMalloc((void **)&d_min, 1 * sizeof(float));
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);
    //ReduceMaxMinPerTensor<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min);
    //GetScaleAndZPSymmetric<float><<<1, 1>>>(d_max, d_min, nums, quantization_bit, d_scale, d_zeropoint);//scale only shape 1
    //QuantizePerTensorSymmetric<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output);
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&milliseconds, start, stop);
    LAUNCH_GPU_KERNEL(ReduceMaxMinPerTensor, QuantizePerTensorSymmetric, 1, nums, HW);
  } else {
  // switch to per channel
    LAUNCH_GPU_KERNEL(ReduceMaxMinPerChannel, QuantizePerChannelSymmetric, channel, channel, HW);
    //cudaMalloc((void **)&d_scale, channel * sizeof(float));
    //cudaMalloc((void **)&d_zeropoint, channel * sizeof(float));
    //cudaMalloc((void **)&d_max, channel * sizeof(float));
    //cudaMalloc((void **)&d_min, channel * sizeof(float));
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);
    //ReduceMaxMinPerChannel<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min, channel, HW);
    //GetScaleAndZPSymmetric<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale, d_zeropoint);
    //QuantizePerChannelSymmetric<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output, channel, HW);
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&milliseconds, start, stop);
  }

  cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost);
  // (per tensor) get CPU output to validate GPU result is right or not
  float* CPUOutput= (float*) malloc(sizeof(float) * nums);
  if(per_tensor_quantize) {
    float* scale = (float*) malloc(sizeof(float) * 1);
    float* zeropoint = (float*) malloc(sizeof(float) * 1);
    GenScalePerTensorSymmetricCPU<float>(input, quantization_bit, nums, scale, zeropoint);
    QuantizationPerTensorSymmetricCPU<float>(input, *scale, quantization_bit, nums, CPUOutput);
    free(scale);
    free(zeropoint);
  } else {
    float* scale = (float*) malloc(sizeof(float) * channel);
    float* zeropoint = (float*) malloc(sizeof(float) * channel);
    GenScalePerChannelSymmetricCPU<float>(input, quantization_bit, HW, channel, nums, scale, zeropoint);
    QuantizationPerChannelSymmetricCPU<float>(input, scale, quantization_bit, HW, nums, CPUOutput);
    free(scale);
    free(zeropoint);
  }
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
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_scale);
  cudaFree(d_zeropoint);
  cudaFree(d_max);
  cudaFree(d_min);
}
