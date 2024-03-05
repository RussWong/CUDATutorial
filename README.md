# CUDATutorial
A CUDA tutorial to make people learn CUDA program from 0

## test enviroment
Turing T4 GPU
## compile command

1. compile by hand

`nvcc xxx.cu -o xxx`

if that does not work, pls try:

`nvcc xxx.cu --gpu-architecture=compute_yy -o xxx`

xxx is file name, yy is GPU compute capability, ep.A100's compute capability is 86.

2. one-click compile and run

please ensure:

1.cmake version >= 3.8

2.you have CUDA TOOLKIT installed in system root directory, downloaded link is https://developer.nvidia.com/cuda-downloads.

```
 mkdir build 
 cd build 
 cmake .. && make -j8 
 cd bin 
 ./xxx
```
## remark
* related performance data is attached at the top of code file.
* the performance data is diverse and diverse on different GPU platforms and NVCC compiler, so some counter-intuitive result is normal, we should only explore and debug the result.
* welcome all comments and pull requests.

## update notes
### v2.0
* add cuda stream
* add quantize
### v2.1
* add fp32/fp16 gemv(vec * mat,mat is col major)
### v2.2
* add fp32/fp16 gemv(vec * mat,mat is row major)
* add some code explaination(WIP)
