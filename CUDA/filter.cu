#include <cstdio>
#include <cuda_runtime.h>
#include "include/CudaAllocator.h"
#include "include/helper_cuda.h"
#include <vector>

__global__ void parallel_filter(int *sum, int *res, int const *arr, int n){
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    if(arr[i] >= 2){
      //old = atomicAdd(dst, src)
      //利用这一点可以实现往一个全局的数组res里追加数据的效果，即push_back,
      //其中sum起到了记录当前数组大小的作用
      int loc = atomicAdd(&sum[0], 1);
      res[loc] = arr[i];
    }
  }
}

int main(){
  int n = 1 << 24;
  std::vector<int, CudaAllocator<int>> arr(n);
  std::vector<int, CudaAllocator<int>> sum(1);
  std::vector<int, CudaAllocator<int>> res(n);

  for(int i = 0; i < n; i ++){
    arr[i] = std::rand() % 4;
  }

  parallel_filter<<<n / 4096, 512>>>(sum.data(), res.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());

  for(int i = 0; i < sum[0]; i ++){
    if(res[i] < 2){
      printf("Wrong At %d\n", i);
      return -1;
    }
  }
  printf("All Correct\n");

  return 0;
}
