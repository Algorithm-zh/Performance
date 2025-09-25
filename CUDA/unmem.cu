#include <cstdio>
#include <cuda_runtime.h>
#include "include/helper_cuda.h"

__global__ void kernal(int *pret){
  *pret = 42;
}


int main(){

  int *pret;
  //um, 统一内存地址，cpu和gpu都可以访问
  //实际上是驱动自动进行的拷贝，省去了自己操作
  //但是会有开销，最好别用
  checkCudaErrors(cudaMallocManaged(&pret, sizeof(int)));
  kernal<<<1, 1>>>(pret);
  checkCudaErrors(cudaDeviceSynchronize());
  printf("result: %d\n", *pret);
  cudaFree(pret);
  return 0;
}
