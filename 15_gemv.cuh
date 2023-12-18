#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <string>
#include <stdexcept>
static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}
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

template<typename T>
struct Vec {
    static constexpr int size = 4;
};
template<>
struct Vec<half> {
    static constexpr int size = 8;
};

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<>
struct SumOp<half> {
  __device__ __forceinline__ half operator()(const half& a, const half& b) const { return __hadd(a, b); }
};

template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31) / 32;
    static __shared__ float warpres[64];
    val = warpReduce<ReductionOp, T>(val);
    if (lane_id == 0){
        warpres[warp_id] = val;
    }
    __syncthreads();
    float warp_val = tid < warp_nums ? warpres[tid] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}

// 一个blk计算一个元素
// mat * vec = {M, N} * {N, 1}/{1, N}
template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(float* matrix, float* vector, float* res, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float thread_local_sum = 0.0f;
    for(int i = 0; i < VECS_PER_THREAD; i++) {
        float4* mat4 = reinterpret_cast<float4*>(&matrix[bid * cols + tid * VEC_SIZE]); // 4 * half2
        float4* vec4 = reinterpret_cast<float4*>(&vector[tid * VEC_SIZE]);
        thread_local_sum += mat4[i].x * vec4[i].x;
        thread_local_sum += mat4[i].y * vec4[i].y;
        thread_local_sum += mat4[i].z * vec4[i].z;
        thread_local_sum += mat4[i].w * vec4[i].w;
    }
    //reduce to get the final val
    float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    //store to gmem
    if(tid == 0) {
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}

struct half8 {
    half2 x;
    half2 y;
    half2 w;
    half2 z;
};

template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(half* matrix, half* vector, half* res, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    //float thread_local_sum = 0.0f;
    half thread_local_sum = 0;
    for(int i = 0; i < VECS_PER_THREAD; i++) {
        float4* mat4 = reinterpret_cast<float4*>(&matrix[bid * cols + tid * VEC_SIZE]); // 4 * half2
        float4* vec4 = reinterpret_cast<float4*>(&vector[tid * VEC_SIZE]);
        half2* vec_h1 = (half2*)&vec4[i].x;
        half2* vec_h2 = (half2*)&vec4[i].y;
        half2* vec_h3 = (half2*)&vec4[i].z;
        half2* vec_h4 = (half2*)&vec4[i].w;
        half2* mat_h1 = (half2*)&mat4[i].x;
        half2* mat_h2 = (half2*)&mat4[i].y;
        half2* mat_h3 = (half2*)&mat4[i].z;
        half2* mat_h4 = (half2*)&mat4[i].w;   
        half2 res1 = __hmul2(*mat_h1, *vec_h1);
        half2 res2 = __hmul2(*mat_h2, *vec_h2);
        half2 res3 = __hmul2(*mat_h3, *vec_h3);
        half2 res4 = __hmul2(*mat_h4, *vec_h4); 
        half2 res = __hadd2(__hadd2(__hadd2(res1, res2), res3), res4);
        thread_local_sum = __hadd(res.x, res.y);
        // float2 res1 = __half22float2(__hmul2(*mat_h1, *vec_h1));
        // float2 res2 = __half22float2(__hmul2(*mat_h2, *vec_h2));
        // float2 res3 = __half22float2(__hmul2(*mat_h3, *vec_h3));
        // float2 res4 = __half22float2(__hmul2(*mat_h4, *vec_h4));
        // thread_local_sum += res1.x;
        // thread_local_sum += res1.y;
        // thread_local_sum += res2.x;
        // thread_local_sum += res2.y;
        // thread_local_sum += res3.x;
        // thread_local_sum += res3.y;
        // thread_local_sum += res4.x;
        // thread_local_sum += res4.y;
        if(i == 0 && tid == 0 && bid == 0) {
            printf("thread sum = %f\n", (float)thread_local_sum); // 8
            // printf("res1.x = %f\n", res1.x); // 1
            // printf("res1.y = %f\n", res1.y);
        }
    }
    //reduce to get the final val
    half reduce_res = blockReduce<SumOp, half>(thread_local_sum);
    // float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    //store to gmem
    if(tid == 0) {
        printf("block reduce_res = %f\n", (float)reduce_res);
        // res[blockIdx.x] = __float2half(reduce_res);
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}


template<int VECS_PER_THREAD, int VEC_SIZE, int THREAD_NUMS>
struct DispatchLauncher
{
    template<typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N){
        dim3 Grid(M);
        dim3 Block(THREAD_NUMS);
        float milliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        gemv<VECS_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[TM][ERROR] CUDA runtime error: ") +  (_cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
};
