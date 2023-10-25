#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
// bug1:长时间运行无结果
// bug2:从某个index开始，cpu和gpu都为0
typedef float FLOAT;

/* CUDA kernel function */
__global__ void vec_add(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    /* 1D grid */
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x){
    	z[i] = y[i] + x[i];
      if(i==500) printf("index500,gpuz=%f,y=%f,x=%f\n",z[i],y[i],x[i]);
	  }
}

void vec_add_cpu(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    for (int i = 0; i < N; i++) {
        z[i] = y[i] + x[i];
        if(i==500) printf("i=500,z=%f\n",z[i]);
    }
}

int main()
{
    int N = 10000;
    int nbytes = N * sizeof(FLOAT);
    int nstreams = 1;// 5:500,2:1250,1:2500
    int size_per_stream = N / nstreams;// assert N can be exactly divided by nstream

    /* 1D block */
    int bs = 256;

    /* 1D grid */
    int s = ceil((size_per_stream + bs - 1.) / bs);
    dim3 grid(s);

    FLOAT *dx, *hx;
    FLOAT *dy, *hy;
    FLOAT *dz, *hz;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);
    
    /* init time */
    float milliseconds = 0;

    /* !must alllocate CPU pinned mem using cudaMallocHost*/
    cudaHostAlloc(&hx, nbytes, cudaHostAllocDefault);
    cudaHostAlloc(&hy, nbytes, cudaHostAllocDefault);
    cudaHostAlloc(&hz, nbytes, cudaHostAllocDefault);

    /* init */
    for (int i = 0; i < N; i++) {
        hx[i] = 1.0;
        hy[i] = 1.0;
    }
    cudaStream_t streams[nstreams];
    
    for (int i = 0; i < nstreams; i++) {
		    cudaStreamCreate(&streams[i]);
        printf("creating %d th stream\n", i);
    }
    for(int i = 0; i < nstreams; i++){
        printf("%d th stream is working \n", i);
        int start_per_stream = i * size_per_stream;
        printf("size_per_steram=%d, start_per_stream=%d\n",size_per_stream,start_per_stream);
        /* async copy data to GPU */
        cudaMemcpyAsync(dx + start_per_stream, hx + start_per_stream, 
                    size_per_stream, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(dy + start_per_stream, hy + start_per_stream, 
                    size_per_stream, cudaMemcpyHostToDevice, streams[i]);

        /* launch GPU kernel */
        vec_add<<<grid, bs, 0, streams[i]>>>(dx + start_per_stream, dy + start_per_stream, dz + start_per_stream, size_per_stream); 
        
        /* async copy GPU result to CPU */
        cudaMemcpyAsync(hz + start_per_stream, dz + start_per_stream, size_per_stream, cudaMemcpyDeviceToHost, streams[i]);
	  } 
    cudaDeviceSynchronize();
    /* CPU compute */
    FLOAT* hz_cpu_res = (FLOAT *) malloc(nbytes);
    vec_add_cpu(hx, hy, hz_cpu_res, N);

    printf("N=%d\n", N);
    /* check GPU result with CPU*/
    for (int i = 0; i < N; ++i) {
        if(i==500) printf("i=500,cpu=%f,gpu=%f\n",hz_cpu_res[i],hz[i]);
        if (fabs(hz_cpu_res[i] - hz[i]) > 1e-6) {
            printf("index: %d, cpu: %f, gpu: %f\n", i, hz_cpu_res[i], hz[i]);
            break;
            //printf("Result verification failed at element index %d!\n", i);
        }
    }
    printf("Result right\n");
    printf("Mem BW= %f (GB/sec)\n", (float)N*4/milliseconds/1e6);
    for (int i = 0; i < nstreams; i++) {
		    cudaStreamDestroy(streams[i]);
        printf("destroying %d th stream\n", i);
    }
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    // free pinned memory
    cudaFreeHost(hx);
    cudaFreeHost(hy);
    cudaFreeHost(hz);
    free(hz_cpu_res);

    return 0;
}
