#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
// 注: 每个cuda程序(.cu文件)的main函数功能大同小异，后面不会每个cu文件都注释main函数逻辑
//999ms
__global__ void reduce_baseline(const int* input, int* output, size_t n) {
  // 由于只分配了1个block和thread,此时cuda程序相当于串行程序
  int sum = 0;
  // 累加
  for (size_t i = 0; i < n; ++i) {
    sum += input[i];
  }
  // 累加结果写回显存
  *output = sum;
}

bool CheckResult(int *out, int groudtruth, int n){
    if (*out != groudtruth) {
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
    //const int blockSize = 256;
    const int blockSize = 1;
    //int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);//used later
    int GridSize = 1;
    // 分配内存和显存并初始化数据
    int *a = (int *)malloc(N * sizeof(int));
    int *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));

    int *out = (int*)malloc((GridSize) * sizeof(int));
    int *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(int));

    for(int i = 0; i < N; i++){
        a[i] = 1;
    }

    int groudtruth = N * 1;
    // 把初始化后的数据拷贝到GPU
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    // 定义分配的block数量和threads数量
    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // 分配1个block和1个thread
    reduce_baseline<<<1, 1>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // 将结果拷回CPU并check正确性
    cudaMemcpy(out, d_out, GridSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < GridSize;i++){
            printf("res per block : %lf ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_baseline latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}