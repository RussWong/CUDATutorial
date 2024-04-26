#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

// v2: 消除shared memory bank conflict
// latency: 2.300ms
template<int blockSize>
__global__ void reduce_v2(float *d_in,float *d_out){
    __shared__ float smem[blockSize];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[gtid];
    __syncthreads();

    // 基于v1作出改进: 从之前的当前线程ID加2*线程ID位置然后不断加上*2位置上的数据，改成不断地对半相加，以消除bank conflict
    // 此时一个block对d_in这块数据的reduce sum结果保存在id为0的线程上面
    for (unsigned int index = blockDim.x / 2; index > 0; index >>= 1) {
        if (tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // store: 哪里来回哪里去，把reduce结果写回显存
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
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
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v2<blockSize><<<Grid,Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        //for(int i = 0; i < GridSize;i++){
            //printf("res per block : %lf ",out[i]);
        //}
        //printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v2 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
