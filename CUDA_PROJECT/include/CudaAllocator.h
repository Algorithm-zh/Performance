#pragma once
#include <cstdio>
#include <utility>
#include "helper_cuda.h"

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
