#include <cstdio>
#include <cuda_runtime.h>
#include "include/helper_cuda.h"

__global__ void kernal(int *arr, int n){
  int i = threadIdx.x + blockDim.x * blockIdx.x; 
  if(i >= n)return;
  arr[i] = i;
}

int main(){

  int n = 32;
  int *arr;
  checkCudaErrors(cudaMallocManaged(&arr, sizeof(arr) * n));
  int nthreads = 4;
  dim3 grim(nthreads);
  //向上整除的方式，防止漏掉元素
  dim3 block((n + nthreads + 1) / nthreads);
  kernal<<<grim, block>>>(arr, n);
  checkCudaErrors(cudaDeviceSynchronize());
  for(int i = 0; i < n; i ++){
    printf("arr[%d]: %d\n", i, arr[i]);
  }
  cudaFree(arr);
  
  return 0;
}
