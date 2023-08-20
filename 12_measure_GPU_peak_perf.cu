#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#define LOOP_TIMES 1000
//T4 fp32: 8.08 TFLOPS
__global__ void FP32FLOPS(int* start, int* stop, float* x, float* y, float* result) {
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    float d1 = x[gtid];
    float d2 = y[gtid];
    float res = 0;
    int start_time = 0;
    // only measure the computation time, eliminate the memory access time
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start_time) :: "memory");
    // Q1: why use 4 fma instruction to get GPU peak performance?
    // A1: we use >2(3or4) fma instruction to hide for loop comparsion and addition instruction overhead
    // Q2: why use 4 dependant fma instruction to get GPU peak performance, can we use 4 independant ones?
    // A2: yes, we can use 2/3/4 independant ones
    for (int i = 0; i < LOOP_TIMES; i++) {
        //asm volatile ("{\n\t""fma.rn.f32 %0, %1, %2 , %0; \n\t"
        //                     "fma.rn.f32 %0, %1, %2 , %0; \n\t"
        //                     "fma.rn.f32 %0, %1, %2 , %0; \n\t"
        //                     "fma.rn.f32 %0, %1, %2 , %0; \n\t"
        //                     "}" : "+f"(res), "+f"(d1),"+f"(d2)); // res + d1 * d2 = res
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
    }
    asm volatile("bar.sync 0;");//sync all threads

    int stop_time = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop_time) :: "memory");
    start[gtid] = start_time;
    stop[gtid] = stop_time;
    result[gtid] = res;
}

int main() {
    int N = 1024;
    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));
    float *d_x;
    float *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float)); 
    cudaMalloc((void **)&d_y, N * sizeof(float)); 
    for(int i = 0; i < 1024; i++) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    float *d_result;
    int *startClock = (int*)malloc(N * sizeof(int));
    int *stopClock = (int*)malloc(N * sizeof(int));
    int *d_startClock;
    int *d_stopClock;
    cudaMalloc((void **)&d_result, N * sizeof(float)); 
    cudaMalloc((void **)&d_startClock, N * sizeof(int)); 
    cudaMalloc((void **)&d_stopClock, N * sizeof(int)); 
    // confirm launch max threads of SM = 1024 to do FMA to saturate SM resource
    FP32FLOPS<<<1, 1024>>>(d_startClock, d_stopClock, d_x, d_y, d_result);
    cudaMemcpy(startClock, d_startClock, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClock, d_stopClock, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    
    int ThreadsPerSM = props.maxThreadsPerMultiProcessor;
    float FLOPS = (LOOP_TIMES * 4 * 2 * 1024) / (static_cast<float>(stopClock[0] - startClock[0]));
    printf( "  GPU Max Clock rate: %0.2f GHz\n" , props.clockRate * 1e-6f);
    printf(" SM counts is %d", props.multiProcessorCount);
    printf("actual NVIDIA T4 GPU peak FLOPS is %f (TFLOPS) \n", FLOPS * props.clockRate * 1e-9 * props.multiProcessorCount);
    free(x);
    free(y);
    free(startClock);
    free(stopClock);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    cudaFree(d_startClock);
    cudaFree(d_stopClock);
}
