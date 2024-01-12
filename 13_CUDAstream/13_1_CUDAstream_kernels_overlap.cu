#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void float_add_one(float* buffer, uint32_t n)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = gid; i < n; i += stride)
    {
        buffer[i] += 1.0f;
    }
}

void float_add_one_launcher(float* buffer, uint32_t n, dim3& threads_per_block,
                          dim3& blocks_per_grid, cudaStream_t stream)
{
    float_add_one<<<blocks_per_grid, threads_per_block, 0, stream>>>(buffer, n);
}

int main(int argc, char** argv)
{
    const size_t buffer_size = 1024 * 10240;
    const size_t num_streams = 5;

    dim3 threads_per_block(1024);
    // Try different values for blocks_per_grid and see stream result on nsight.
    // 1, 2, 4, 8, 16, 32, 1024
    dim3 blocks_per_grid(32);
    
    std::vector<float*> d_buffers(num_streams);
    // below can be replaced by cudaStream_t*;
    // cudaStream_t* streams = (cudaStream_t*) malloc(num_streams * sizeof(cudaStream_t));
    std::vector<cudaStream_t> streams(num_streams);

    for (auto& d_buffer : d_buffers)
    {
        cudaMalloc(&d_buffer, buffer_size * sizeof(float));
    }

    for (auto& stream : streams)
    {
        cudaStreamCreate(&stream);
    }
    // each independent kernel running on each streams to parallize
    for (int i = 0; i < num_streams; ++i)
    {
        float_add_one_launcher(d_buffers[i], buffer_size, threads_per_block,
                             blocks_per_grid, streams[i]);
    }
    // host wait each stream to sync
    for (auto& stream : streams)
    {
        cudaStreamSynchronize(stream);
    }

    for (auto& d_buffer : d_buffers)
    {
        cudaFree(d_buffer);
    }

    for (auto& stream : streams)
    {
        cudaStreamDestroy(stream);
    }

    return 0;
}
