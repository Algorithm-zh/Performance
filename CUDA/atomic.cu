#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "include/CudaAllocator.h"
#include "include/helper_cuda.h"
#include <vector>

__global__ void parallel_sum(int *sum, int const *arr, int n){
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    atomicAdd(&sum[0], arr[i]);
    //而不是用下面这个,因为并行编程要保证原子操作
    //sum[0] += arr[i];
  }
}

int main(){

  int n = 65536;
  std::vector<int, CudaAllocator<int>> arr(n);
  //为什么这么写sum,因为你直接写是在cpu上的，所以这样写了个在gpu上分配的大小为1的vector
  std::vector<int, CudaAllocator<int>> sum(1);

  for(int i = 0; i < n; i ++){
    arr[i] = std::rand() % 4;
  }

  parallel_sum<<<n / 128, 128>>>(sum.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());

  printf("result: %d\n", sum[0]);
  return 0;
}

