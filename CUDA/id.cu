#include <cstdio>
#include <cuda_runtime.h>

__global__ void another(){
  printf("another: Thread %d of %d\n", threadIdx.x, blockDim.x);
}

__global__ void kernal(){
  //线程块id, 线程块数量，局部线程id, 当前块线程数量
  printf("Block %d of %d, Thread %d of %d\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);

  //扁平化后的id
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tnum = blockDim.x * gridDim.x;
  printf("Flattened Thread %d of %d\n", tid, tnum);

  //从kelper架构开始，global里可以调另一个global
  // another<<<dim3(2, 1, 1), dim3(3, 1, 1)>
}



int main(){

  //一共两个线程块，每个线程块里有三个线程
  kernal<<<2, 3>>>();
  cudaDeviceSynchronize();
  return 0;
}
