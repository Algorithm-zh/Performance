#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "include/helper_cuda.h"
#include "include/CudaAllocator.h"
#include <vector>

template <class Func>
__global__ void parallel_for(int n, Func func){
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    func(i);
  }
}

int main(){
  int n = 65536;
  std::vector<float, CudaAllocator<float>> arr(n);

  parallel_for<<<32, 128>>>(n, [arr = arr.data()]__device__(int i){
    arr[i] = sinf(i);//针对float,普通的是针对double,会影响性能
  });

  checkCudaErrors(cudaDeviceSynchronize());
  for(int i = 0; i < n; i ++){
    printf("diff = %f\n", arr[i] - sinf(i));
  }

  return 0;
}
