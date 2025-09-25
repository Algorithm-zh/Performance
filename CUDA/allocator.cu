#include <cstdio>
#include <cuda_runtime.h>
#include "include/helper_cuda.h"
#include <vector>

template <class T>
struct CudaAllocator{
  using value_type = T;

  T *allocate(size_t size){
    T *ptr = nullptr;
    checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
    return ptr;
  }

  void deallocate(T *ptr, size_t size = 0){
    checkCudaErrors(cudaFree(ptr));
  }

  //可以通过给allocator添加construct成员函数，魔改vector对元素的构造，默认情况下可以有任意多个参数，而如果没有参数则说明无参构造函数
  //因此只需要判断是不是有参数，然后判断是不是传统的c语言类型，如果是则跳过无参构造，避免在cpu上进行零初始化
  template <class ...Args>
  void construct(T *p, Args &&...args){
    if constexpr (!(sizeof...(Args) == 0 && std::is_standard_layout_v<T>))
      ::new((void *)p) T(std::forward<Args>(args)...);
  }
};

__global__ void kernal(int *arr, int n){
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    arr[i] = i;
  }
}

int main(){
  int n = 65536;
  //vector会调用std::allocator<T>的allocate/deallocate成员函数，里面调用的标准库的malloc/free,即是cpu内存
  //我们可以自己定义一个std::allocator<T>一样具有allocate/deallocate成员函数的类，就能让vector在cuda分配内存
  std::vector<int, CudaAllocator<int>> arr(n);

  kernal<<<32, 128>>>(arr.data(), n);

  checkCudaErrors(cudaDeviceSynchronize());
  for(int i = 0; i < n; i ++){
    printf("arr[%d]: %d\n", i, arr[i]);
  }
  return 0;
}

