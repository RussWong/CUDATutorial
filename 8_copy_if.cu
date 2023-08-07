#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "cooperative_groups.h"
//#define THREAD_PER_BLOCK 256
//估计这种warp和shared在老的gpu上面会很有成效，但是在turing后的GPU，nvcc编译器优化了很多
//cpu
int filter(int *dst, int *src, int n) {
  int nres = 0;
  for (int i = 0; i < n; i++)
    if (src[i] > 0)
      dst[nres++] = src[i];
  // return the number of elements copied
  return nres;
}
//数据量为256000000时，latency=1.437ms
//cuda naive kernel
//__global__ void filter_k(int *dst, int *nres, int *src, int n) {
//  int i = threadIdx.x + blockIdx.x * blockDim.x;
//  if(i < n && src[i] > 0)
//    dst[atomicAdd(nres, 1)] = src[i];
//}
//数据量为256000000时，latency=1.389ms
// //block level, use block level atomics based on shared memory
// __global__ 
// void filter_shared_k(int *dst, int *nres, const int* src, int n) {
//   __shared__ int l_n;
//   int gtid = blockIdx.x * blockDim.x + threadIdx.x;
//   int total_thread_num = blockDim.x * gridDim.x;

//   for (int i = gtid; i < n; i += total_thread_num) {
//     // use first thread to zero the counter
//     if (threadIdx.x == 0)
//       l_n = 0;
//     __syncthreads();

//     // 每个block内部，大于0的数量(l_n)和每个大于0的thread offset(pos)
//     int d, pos;

//     if(i < n) {
//       d = src[i];
//       if(d > 0)
//         //pos: src[thread]>0的thread在当前block的index
//         pos = atomicAdd(&l_n, 1);
//     }
//     __syncthreads();

//     //每个block选出tid=0作为leader
//     //leader把每个block的大于0的数量l_n累加到 the global counter(nres)
//     if(threadIdx.x == 0)
//       l_n = atomicAdd(nres, l_n);
//     __syncthreads();

//     //write & store
//     if(i < n && d > 0) {
//     //pos: src[thread]>0的thread在当前block的index
//     //l_n: 在当前block的前面几个block的所有src>0的个数
//     //pos + l_n：当前thread的全局offset
//       pos += l_n; 
//       dst[pos] = d;
//     }
//     __syncthreads();
//   }
// }
//数据量为256000000时，latency=1.389ms
//warp level, use warp-aggregated atomics
__device__ int atomicAggInc(int *ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active);//warp mask中为1的数量
  int lane_mask_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
  unsigned int rank = __popc(active & lane_mask_lt);//比当前线程id小且值为1的mask之和
  int warp_res;
  if(rank == 0)//leader thread
    warp_res = atomicAdd(ctr, change);//compute global offset of warp
  warp_res = __shfl_sync(active, warp_res, leader);//broadcast to every thread
  return warp_res + rank;
}
//0.157ms,14.373ms(256000000 data)
__global__ void filter_warp_k(int *dst, int *nres, const int *src, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i >= n)
    return;
  if(src[i] > 0)
    dst[atomicAggInc(nres)] = src[i];
}

bool CheckResult(int *out, int groudtruth, int n){
    //for (int i = 0; i < n; i++){
    if (*out != groudtruth) {
        return false;
    }
    //}
    return true;
}

int main(){
    float milliseconds = 0;
    int N = 2560000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

    int *src_h = (int *)malloc(N * sizeof(int));
    int *dst_h = (int *)malloc(N * sizeof(int));
    int *nres_h = (int *)malloc(1 * sizeof(int));
    int *dst, *nres;
    int *src;
    cudaMalloc((void **)&src, N * sizeof(int));
    cudaMalloc((void **)&dst, N * sizeof(int));
    cudaMalloc((void **)&nres, 1 * sizeof(int));

    for(int i = 0; i < N; i++){
        src_h[i] = 1;
    }

    int groudtruth = 0;
    for(int j = 0; j < N; j++){
        if (src_h[j] > 0) {
            groudtruth += 1;
        }
    }


    cudaMemcpy(src, src_h, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    filter_warp_k<<<Grid, Block>>>(dst, nres, src, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(nres_h, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("%lf ",*nres_h);
        printf("\n");
    }
    printf("filter_k latency = %f ms\n", milliseconds);    

    cudaFree(src);
    cudaFree(dst);
    cudaFree(nres);
    free(src_h);
    free(dst_h);
    free(nres_h);
}
