#include <cuda.h>
#include <bits/stdc++.h>
#include "cuda_runtime.h"

// 实现fp32的fused biasadd mask scale and add的融合算子
// biasadd + mask + scale + elemwise_add四个算子的融合
// （x + bias） * mask * scale + addend;

template<typename T>
struct MaskScaleAndElemwiseAddFunctor
{
    // 有参构造函数
    MaskScaleAndElemwiseAddFunctor(const uint8_t * mask, const T * add_val, float scale)
    :_mask(mask), _add_val(add_val), _scale(scale)
    {}

    // 重载运算符（）
    __device__ T operator()(T x, int i) const
    {
        return x * static_cast<T>(static_cast<bool>(_mask[i]) * _scale) + _add_val[i];
    }

    const uint8_t * _mask;
    const T * _add_val;
    float _scale;
};

template<int biasSize, typename FUNCTOR, typename T>
__global__ void FusedBaisAdd(FUNCTOR functor, T * dx, T * dy, T * d_bias, const int n, const int bias_size)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = gid; i < n; i += gridDim.x * blockDim.x)
    {
        T tmp = dx[i] + d_bias[i % bias_size];
        dy[i] = functor(tmp, i);
    }
}

// 使用向量化进行存取
template<int biasSize, typename FUNCTOR, typename T>
__global__ void FusedBaisAddVecSmem(FUNCTOR functor, T * dx, T * dy, T * d_bias, const int n, const int bias_size)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ T smem[biasSize];

    // 将d_bias放在shared memory上
    if (tid < bias_size)
        smem[tid] = d_bias[tid];
    __syncthreads();

    for (int i = gid; i < n / 4; i += gridDim.x * blockDim.x)
    {
        float4 a = reinterpret_cast<float4 *>(dx)[i];
        float4 b;

        b.x = functor(a.x + smem[(i * 4) % bias_size], i * 4);
        b.y = functor(a.y + smem[(i * 4 + 1) % bias_size], i * 4 + 1);
        b.z = functor(a.z + smem[(i * 4 + 2) % bias_size], i * 4 + 2);
        b.w = functor(a.w + smem[(i * 4 + 3) % bias_size], i * 4 + 3);

        reinterpret_cast<float4*>(dy)[i] = b;
    }
}

bool CheckRight(float * y, float * groudTruth, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (y[i] != groudTruth[i])
        {
            printf("y[%d] : %f \n", i, y[i]);
            printf("groundTruth[%d] : %f\n", i, groudTruth[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    constexpr int n = 100000;
    constexpr int bias_size = 10;
    
    float scale = 0.5;
    uint8_t * mask_tensor = new uint8_t[n];
    float * add_val = new float[n];
    // 初始化
    for (int i = 0; i < n; ++i)
    {
        mask_tensor[i] = (uint8_t)(i);
        add_val[i] = (float)(i);
    }

    float * x = (float *)malloc(sizeof(float) * n);
    float * y = (float *)malloc(sizeof(float) * n);
    float * bias = (float *)malloc(sizeof(float) * bias_size);
    for (int i = 0; i < n; ++i)
    {
        x[i] = (float)(i);
        y[i] = 0.0f;
    }
    for (int i = 0; i < bias_size; ++i)
        bias[i] = i;

    float * groudTruth = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i)
    {
        groudTruth[i] = (x[i] + bias[i % bias_size]) * static_cast<float>(static_cast<bool>(mask_tensor[i]) * scale) + add_val[i];
    }

    float * dx, * dy, * d_bias;
    cudaMalloc((void **)&dx, sizeof(float) * n);
    cudaMalloc((void **)&dy, sizeof(float) * n);
    cudaMalloc((void **)&d_bias, sizeof(float) * bias_size);
    cudaMemcpy(dx, x, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * bias_size, cudaMemcpyHostToDevice);
    uint8_t * d_mask_tensor;
    float * d_add_val;
    cudaMalloc((void **)&d_mask_tensor, sizeof(uint8_t) * n);
    cudaMalloc((void **)&d_add_val, sizeof(float) * n);
    cudaMemcpy(d_mask_tensor, mask_tensor, sizeof(uint8_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_add_val, add_val, sizeof(float) * n, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = 512;
    int gridSize = std::min((n + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);

    MaskScaleAndElemwiseAddFunctor<float> functor(d_mask_tensor, d_add_val, scale);

    dim3 Block(blockSize);
    dim3 Grid(gridSize);

    float milliseconds = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int i = 0; i < 1000; ++i)
        FusedBaisAdd<bias_size><<<Grid, Block>>>(functor, dx, dy, d_bias, n, bias_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(y, dy, sizeof(float) * n, cudaMemcpyDeviceToHost);

    bool isRight = CheckRight(y, groudTruth, n);
    if (isRight)
        printf("结果正确\n");
    else
        printf("结果错误\n");    

    printf("it costs %f s \n", milliseconds/1000);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(d_bias);
    cudaFree(d_add_val);
    cudaFree(d_mask_tensor);
    free(x);
    free(y);
    free(bias);
    free(groudTruth);
    delete mask_tensor;
    mask_tensor = nullptr;
    delete add_val;
    add_val = nullptr;
}