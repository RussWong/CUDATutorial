#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"


//latency: 0.694ms
__device__ void WarpReduce(volatile float* sdata, unsigned int tid){
    float x = sdata[tid];
    if (blockDim.x >= 64) {
      x += sdata[tid + 32]; __syncwarp();
      sdata[tid] = x; __syncwarp();
    }
    x += sdata[tid + 16]; __syncwarp();
    sdata[tid] = x; __syncwarp();
    x += sdata[tid + 8]; __syncwarp();
    sdata[tid] = x; __syncwarp();
    x += sdata[tid + 4]; __syncwarp();
    sdata[tid] = x; __syncwarp();
    x += sdata[tid + 2]; __syncwarp();
    sdata[tid] = x; __syncwarp();
    x += sdata[tid + 1]; __syncwarp();
    sdata[tid] = x; __syncwarp();
}

template<int blockSize>
__global__ void reduce_v4(float *d_in,float *d_out){
    __shared__ float sdata[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    // load: 每个线程加载两个元素到shared mem对应位置
    sdata[tid] = d_in[i] + d_in[i + blockSize];
    __syncthreads();

    // compute: reduce in shared mem
    // 思考这里是如何并行的
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // last warp拎出来单独作reduce
    if (tid < 32) {
        WarpReduce(sdata, tid);
    }
    // store: write back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    //printf("%f", res);
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    //const int N = 32 * 1024 * 1024;
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
    dim3 Block(blockSize / 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v4<blockSize / 2><<<Grid,Block>>>(d_a, d_out);
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
    printf("reduce_v4 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}