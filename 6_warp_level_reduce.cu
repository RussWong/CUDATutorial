//warp shuffle
#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define WarpSize 32
//latency: 1.254ms
template <int blockSize>
__device__ float WarpShuffle(float sum) {
    //__shfl_down_sync：前面的thread向后面的thread要数据
    //__shfl_up_sync: 后面的thread向前面的thread要数据
    //返回前面的thread向后面的thread要的数据，比如__shfl_down_sync(0xffffffff, sum, 16)那就是返回16号线程，17号线程的数据
    //warp内的数据交换不会出现warp在shared memory上交换数据时的不一致现象，无需syncwarp
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <int blockSize>
__global__ void reduce_warp_level(float *d_in,float *d_out, unsigned int n){
    float sum = 0;//当前线程的私有寄存器，即每个线程都会拥有一个sum寄存器

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;
    unsigned int total_thread_num = blockSize * gridDim.x;

    for (int i = gtid; i < n; i += total_thread_num)
    {
        sum += d_in[i];//thread local reduce，一个block/thread处理多个元素
    }
    
    // partial sums for each warp
    __shared__ float WarpSums[blockSize / WarpSize]; 
    const int laneId = tid % WarpSize;
    const int warpId = tid / WarpSize; 
    sum = WarpShuffle<blockSize>(sum);
    if(laneId == 0) {
        WarpSums[warpId] = sum;
    }
    __syncthreads();
    //至此，得到了每个warp的reduce sum结果
    //接下来，再使用第一个warp(laneId=0-31)对每个warp的reduce sum结果求和
    //首先，把warpsums存入前blockDim.x / WarpSize个线程的sum寄存器中
    //接着，继续warpshuffle
    sum = (tid < blockSize / WarpSize) ? WarpSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0) {
        sum = WarpShuffle<blockSize/WarpSize>(sum); 
    }
    // write result for this block to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sum;
    }
}

bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_warp_level<blockSize><<<Grid,Block>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d \n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < GridSize;i++){
            printf("resPerBlock : %lf ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_warp_level latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
