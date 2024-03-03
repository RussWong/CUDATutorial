#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 100000000   //Array size has to exceed L2 size to avoid L2 cache residence
#define MEMORY_OFFSET 10000000
#define BENCH_ITER 10
#define THREADS_NUM 256

__device__ __forceinline__
float4 LoadFromGlobalPTX(float4 *ptr) {
    float4 ret;
    // ptx指令，是CUDA的更底层的语言，类似于汇编对于C/C++
    asm volatile (
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
        : "l"(ptr)
    );

    return ret;
}

//float4 vectoradd
__global__ void mem_bw (float* A,  float* B, float* C){
    // 泛指当前线程在所有block范围内的全局id
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	for(int i = idx; i < MEMORY_OFFSET / 4; i += blockDim.x * gridDim.x) {
		//问题1: 删除43-46行,会发现带宽数据为2666g/S
		//尝试: 使用nv ptx load global memory指令,结果数据依然没变
		//结论: 大概率是编译器优化:读了数据不做操作那就会不读
		float4 a1 = reinterpret_cast<float4*>(A)[i];
		//float4 a1 = LoadFromGlobalPTX(reinterpret_cast<float4*>(A) + i);
		float4 b1 = reinterpret_cast<float4*>(B)[i];
		//float4 b1 = LoadFromGlobalPTX(reinterpret_cast<float4*>(B) + i);
		float4 c1;
    		// 测量显存带宽方法1:向量加法,248.8g/s
		//c1.x = a1.x + b1.x;
		//c1.y = a1.y + b1.y;
		//c1.z = a1.z + b1.z;
		//c1.w = a1.w + b1.w;
		// 测量显存带宽方法2:copy操作,242.3g/s	
		c1.x = a1.x;
		c1.y = a1.y;
		c1.z = a1.z;
		c1.w = a1.w;
		reinterpret_cast<float4*>(C)[i] = c1;
	}
}

void vec_add_cpu(float *x, float *y, float *z, int N)
{
    for (int i = 0; i < 20; i++) z[i] = y[i] + x[i];
}

int main(){
	float *A = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *B = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *C = (float*) malloc(ARRAY_SIZE*sizeof(float));

	float *A_g;
	float *B_g;
	float *C_g;

	float milliseconds = 0;

	for (uint32_t i=0; i<ARRAY_SIZE; i++){
		A[i] = (float)i;
		B[i] = (float)i;
	}
	cudaMalloc((void**)&A_g, ARRAY_SIZE*sizeof(float));
	cudaMalloc((void**)&B_g, ARRAY_SIZE*sizeof(float));
	cudaMalloc((void**)&C_g, ARRAY_SIZE*sizeof(float));

	cudaMemcpy(A_g, A, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_g, B, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  
	int BlockNums = MEMORY_OFFSET / 256;
    //warm up to occupy L2 cache
	printf("warm up start\n");
	mem_bw<<<BlockNums / 4, THREADS_NUM>>>(A_g, B_g, C_g);
	printf("warm up end\n");
    // time start using cudaEvent
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	for (int i = BENCH_ITER - 1; i >= 0; --i) {
		mem_bw<<<BlockNums / 4, THREADS_NUM>>>(A_g + i * MEMORY_OFFSET, B_g + i * MEMORY_OFFSET, C_g + i * MEMORY_OFFSET);
	}
	// time stop using cudaEvent
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(C, C_g, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	/* CPU compute */
	float* C_cpu_res = (float *) malloc(20*sizeof(float));
	vec_add_cpu(A, B, C_cpu_res, ARRAY_SIZE);

	/* check GPU result with CPU*/
	for (int i = 0; i < 20; ++i) {
		/* 测量显存带宽时, 修改C_cpu_res[i]为0 */
		if (fabs(C_cpu_res[i] - C[i]) > 1e-6) {
			printf("Result verification failed at element index %d!\n", i);
		}
	}
	printf("Result right\n");
	unsigned N = ARRAY_SIZE * 4;
	/* 测量显存带宽时, 根据实际读写的数组个数, 指定下行是1/2/3 */
	printf("Mem BW= %f (GB/sec)\n", 3 * (float)N / milliseconds / 1e6);
  	cudaFree(A_g);
  	cudaFree(B_g);
  	cudaFree(C_g);

  	free(A);
  	free(B);
  	free(C);
  	free(C_cpu_res);
}
