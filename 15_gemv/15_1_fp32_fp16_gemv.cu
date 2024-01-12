#include "15_gemv.cuh"

template<typename T>
void gemvCPU(T* mat, T* vec, T* dst, int M, int N){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dst[i] +=  mat[i * N + j] * vec[j];
        }
        if (i < 5){
            printf("cpu res = %f\n", dst[i]);
        }
    }
}
template<typename T>
bool CheckResult(T *out, T* groudtruth, int M){
    for (int i = 0; i < M; i++){
      if(i == 0){
        printf("1st comparsion: %f and %f \n" , out[i], groudtruth[i] );
      }
      if (out[i] != groudtruth[i]) {
        printf("%dth res is wrong: %f and %f \n" , i, out[i], groudtruth[i] );
        return false;
      }
    }
    return true;
}


template<typename T>
void gemv_kernel(T *vec,
                T *d_vec,
                T *mat,
                T *d_mat,
                T *dst,
                T *d_dst){

    constexpr int N = 2048;//256 * 8
    constexpr int M = 256;

//    initialize<T>(vec, d_vec, mat, d_mat, dst, d_dst, M, N);
    vec = (T *)malloc(N * sizeof(T));
    cudaMalloc((void **)&d_vec, N * sizeof(T));

    mat = (T *)malloc(M * N * sizeof(T));
    cudaMalloc((void **)&d_mat, M * N * sizeof(T));

    dst = (T*)malloc(M * sizeof(T));
    cudaMalloc((void **)&d_dst, M * sizeof(T));

    for(int i = 0; i < N; i++){
        vec[i] = (T)1;
    }
    for(int i = 0; i < N * M; i++){
        mat[i] = (T)1;
    }

    // gemvCPU(mat, vec, dst, M, N);

    cudaMemcpy(d_vec, vec, N * sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mat, mat, M * N * sizeof(T), cudaMemcpyHostToDevice);
    constexpr int THREAD_NUMS = 256;
    constexpr int VEC_SIZE = Vec<T>::size;
    constexpr int VECS_PER_THREAD = (N / THREAD_NUMS) / VEC_SIZE; // 1 for half, 2 for fp32
    DispatchLauncher<VECS_PER_THREAD, VEC_SIZE, THREAD_NUMS>::template launcher<T>(d_mat, d_vec, d_dst, M, N);

    CHECK(cudaMemcpy(dst, d_dst, M * sizeof(T), cudaMemcpyDeviceToHost));
    T* groudtruth = (T*)malloc(sizeof(T) * M);
    // gemvCPU(mat, vec, groudtruth, M, N);
    // 注意：此处没有验证fp16的cpu gemv，只是打印fp16 cuda kernel的结果肉眼看了一下
    // 验证fp16的结果的做法是L75传入half类型的输入和模板类型，在L4 gemv cpu函数里面将输入类型强转为fp32即可，因为cpu没有half类型
    float* fp32_mat = reinterpret_cast<float*>(mat);
    float* fp32_vec = reinterpret_cast<float*>(vec);
    float* fp32_groudtruth = reinterpret_cast<float*>(groudtruth);
    gemvCPU<float>(fp32_mat, fp32_vec, fp32_groudtruth, M, N);
    float* fp32_dst = reinterpret_cast<float*>(dst);
    bool is_right = CheckResult(fp32_dst, fp32_groudtruth, M);
    // bool is_right = CheckResult(dst, groudtruth, M);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }
    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_dst);
    free(vec);
    free(mat);
    free(dst);
}
template void gemv_kernel<float>(float*, float*, float*, float*, float*, float*);
template void gemv_kernel<half>(half*, half*, half*, half*, half*, half*);

int main() {
    if(true) {
        float *vec;
        float *d_vec;
        float *mat;
        float *d_mat;
        float *dst;
        float *d_dst;
        gemv_kernel<float>(vec, d_vec, mat, d_mat, dst, d_dst);
    } else {
        half *vec;
        half *d_vec;
        half *mat;
        half *d_mat;
        half *dst;
        half *d_dst;
        gemv_kernel<half>(vec, d_vec, mat, d_mat, dst, d_dst);
    }
}