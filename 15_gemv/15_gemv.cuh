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
// 把block reduce拆分为多个warp reduce来计算
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    // 向上进1，以防分配的线程数量小于32导致warp nums为0
    int warp_nums = (blockDim.x + 31) / 32;
    static __shared__ float warpres[64];
    // block内每个warp reduce的结果，该结果保存在每个warp内的0号线程，所以L65用0号线程写入warp res
    val = warpReduce<ReductionOp, T>(val);
    if (lane_id == 0){
        warpres[warp_id] = val;
    }
    __syncthreads();
    // 最后把每个warp的结果再作一个reduce得到最终一个block的结果
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
        // 注意: 此处读取mat4的代码和视频上不同，视频上错以为是先加offset再强转为float4指针
        // 向量化读取matrix和vector，因为是先转成float4指针再读取，所以注意matrix读取的时候，列数需要除以VEC_SIZE
        float4 mat4 = reinterpret_cast<float4*>(matrix)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid]; // 1 * float4
        float4 vec4 = reinterpret_cast<float4*>(vector)[i * blockDim.x + tid];
        // 向量乘法并累加向量内的4个结果，得到该向量内部的乘加结果
        thread_local_sum += mat4.x * vec4.x;
        thread_local_sum += mat4.y * vec4.y;
        thread_local_sum += mat4.z * vec4.z;
        thread_local_sum += mat4.w * vec4.w;
    }
    // reduce to get the final val
    // 以上仅得到了每个向量的内部乘加结果，故还需要reduce得到matrix的一行乘加vector的最终结果
    float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    // store to gmem
    if(tid == 0) {
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}

template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(half* matrix, half* vector, half* res, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    //float thread_local_sum = 0.0f;
    half thread_local_sum = 0;
    for(int i = 0; i < VECS_PER_THREAD; i++) {
        float4 mat4 = reinterpret_cast<float4*>(matrix)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid]; // 4 * half2
        float4 vec4 = reinterpret_cast<float4*>(vector)[i * blockDim.x + tid];
        // 与fp32的gemv不同点在于，向量宽度由4变为8，满足128bit的CUDA线程最大读写宽度
        // 所以依然可以用float4表示读取的偏移宽度，half也OK，只是CUDA没有half8这个内置类型，需要自定义half8这个struct，见下文190行左右
        // 然后再转成half2，调用half2 intrinsic做计算
        half2* vec_h1 = (half2*)&vec4.x;
        half2* vec_h2 = (half2*)&vec4.y;
        half2* vec_h3 = (half2*)&vec4.z;
        half2* vec_h4 = (half2*)&vec4.w;
        half2* mat_h1 = (half2*)&mat4.x;
        half2* mat_h2 = (half2*)&mat4.y;
        half2* mat_h3 = (half2*)&mat4.z;
        half2* mat_h4 = (half2*)&mat4.w;   
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
        //if(i == 0 && tid == 0 && bid == 0) {
            //printf("thread sum = %f\n", (float)thread_local_sum); // 8
            // printf("res1.x = %f\n", res1.x); // 1
        //}
    }
    //reduce to get the final val
    // 以上仅得到了每个向量的内部乘加结果，故还需要reduce得到matrix的一行乘加vector的最终结果
    half reduce_res = blockReduce<SumOp, half>(thread_local_sum);
    //store to gmem
    if(tid == 0) {
        printf("block reduce_res = %f\n", (float)reduce_res);
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
            throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +  (_cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
};

// vec * mat, mat is row major
// [1, N] * [N, M]
// logits * v
// 有关fp32/fp16 fma和add的各种重载操作
namespace gemv2 {
    struct half8 {
        half2 h1;
        half2 h2;
        half2 h3;
        half2 h4;

        __device__ half8& operator = (half8 h8) {
            h1 = h8.h1;
            h2 = h8.h2;
            h3 = h8.h3;
            h4 = h8.h4;
            return *this;
        }
    };

    template<int M, typename T>
    struct get_threads_per_mat_row {
        static const int value = M * sizeof(T) / 16;
    };

    inline __device__ float add(float a, float b)
    {
        return a + b;
    }

    inline __device__ float4 add(float4 a, float4 b)
    {
        float4 c;
        c.x = gemv2::add(a.x, b.x);
        c.y = gemv2::add(a.y, b.y);
        c.z = gemv2::add(a.z, b.z);
        c.w = gemv2::add(a.w, b.w);
        return c;
    }
    inline __device__ half add(half a, half b)
    {
        //return __hadd(a, b);
        //if use L216, half+half is not really adding, its so weird, which  cause our result is 32, not 256
        return (half)((float)a+(float)b);
    }

    inline __device__ half2 add(half2 a, half2 b)
    {
        half2 res;
        res.x = gemv2::add(a.x, b.x);
        res.y = gemv2::add(a.y, b.y);
        return res;
    }

    inline __device__ half8 add(half8 a, half8 b)
    {
        half8 c;
        c.h1 = gemv2::add(a.h1, b.h1);
        c.h2 = gemv2::add(a.h2, b.h2);
        c.h3 = gemv2::add(a.h3, b.h3);
        c.h4 = gemv2::add(a.h4, b.h4);
        return c;
    }

    inline __device__ half fma(half a, half b, half c)
    {
        // 有的编译器会不认识half intrinsic 例如__hmul或者__hadd，这很奇怪
        // 所以粗暴转成fp32计算再转回fp16
        return __float2half((float)a * (float)b + (float)c);
    }


    inline __device__ half2 fma(half a, half2 b, half2 c)
    {
        half2 res;
        res.x = gemv2::fma(a, b.x, c.x);
        res.y = gemv2::fma(a, b.y, c.y);
        return res;
    }

    inline __device__ half8 fma(half a, half8 b, half8 c)
    {
        half8 d;
        d.h1 = gemv2::fma(a, b.h1, c.h1);
        d.h2 = gemv2::fma(a, b.h2, c.h2);
        d.h3 = gemv2::fma(a, b.h3, c.h3);
        d.h4 = gemv2::fma(a, b.h4, c.h4);
        return d;
    }

    inline __device__ float fma(float a, float b, float c)
    {
        return a * b + c;
    }

    inline __device__ float4 fma(float a, float4 b, float4 c)
    {
        float4 d;
        d.x = gemv2::fma(a, b.x, c.x);
        d.y = gemv2::fma(a, b.y, c.y);
        d.z = gemv2::fma(a, b.z, c.z);
        d.w = gemv2::fma(a, b.w, c.w);
        return d;
    }
} // namespace gemv2

// 1个block处理一个[1, M], 循环处理完[N, M]
// for fp32: <64, M * sizeof(T) / 16 = M / 4, 4>
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv2_kernel(float* matrix, float* vector, float* res, int N, int M) {
    //根据编译期常量获取每个thread处理的行列号
    int tid = threadIdx.x;
    // 每个线程负责数据所在行号
    int mat_o = tid / THREADS_PER_VALUE;
    // 每个线程负责数据所在向量号
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
    // 一个block处理的行数
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    __shared__ float out_smem[512];
    float4 out;
    // 点乘或fma，inter-block循环累加
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);
        float logits = vector[ti];
        // fused mul and add: d = a * b + c
        out = gemv2::fma(logits, mat, out);
    }
    // intra-block二分法相加得最终结果
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();
        if (mat_o < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }
    // 二分法最终结果存在首行，写回显存
    if (mat_o == 0) {
        *reinterpret_cast<float4*>(&res[mat_i]) = out;
    }
}

// for fp16: <64, M * sizeof(T) / 16 = M / 8, 8>
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv2_kernel(half* matrix, half* vector, half* res, int N, int M) {
    int tid = threadIdx.x;
    int mat_o = tid / THREADS_PER_VALUE;
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    __shared__ half out_smem[2048];
    gemv2::half8 out;
    // zero(out);
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        gemv2::half8 mat = *reinterpret_cast<gemv2::half8*>(&matrix[ti * M + mat_i]);
        half logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<gemv2::half8*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();

        if (mat_o < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<gemv2::half8*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }
    if (mat_o == 0) {
        *reinterpret_cast<gemv2::half8*>(&res[mat_i]) = out;
    }
}
// TODO: 修改float4部分为可以泛化表示float4和half8类型的代码, 而后此模板函数可以取代以上fp32和fp16的gemv2
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE, typename T>
__global__ void gemv2_kernel_template(T* matrix, T* vector, T* res, int N, int M) {
    int tid = threadIdx.x;
    int mat_o = tid / THREADS_PER_VALUE;
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    __shared__ T out_smem[512];
    float4 out; //TODO
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);//TODO
        T logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;//TODO
        }
        __syncthreads();
        if (mat_o < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);//TODO
        }
        __syncthreads();
    }
    if (mat_o == 0) {
        *reinterpret_cast<float4*>(&res[mat_i]) = out;//TODO
    }
}

template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
struct DispatchLauncher2
{
    template<typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N){
        dim3 Grid(1);
        dim3 Block(THREADS_PER_BLOCK);
        float milliseconds = 0;
        // 使用cudaevent计时，开销最小
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        // 启动cuda kernel
        gemv2_kernel<THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N, M);
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +  (_cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
};
