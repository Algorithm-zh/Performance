#include <stdio.h>
#include <vector>
#include "include/helper_cuda.h"
#include "include/CudaAllocator.h"

//1.核函数支持模板
template <int N, class T>
__global__ void kernal(T *arr){
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x){
    arr[i] = i;
  }
}

//2.核函数支持函数式编程，接受仿函数
template <class Func>
__global__ void parallel_for(int n, Func func){
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x  * gridDim.x){
    func(i);
  }
}

struct MyFunctor{
  __device__ void operator()(int i) const {
    printf("number %d\n", i);
  }
};

int main(){
  constexpr int n = 65536;
  std::vector<int, CudaAllocator<int>> arr(n);

  kernal<n><<<32, 128>>>(arr.data());
  checkCudaErrors(cudaDeviceSynchronize());

  parallel_for<<<32, 128>>>(n, MyFunctor{});
  checkCudaErrors(cudaDeviceSynchronize());

//3.核函数甚至支持lambda表达式，但是必须在cmake里开启--extended-lambda
  //这里先通过data获取数据在gpu上的原始指针，然后对这个原始指针进行浅拷贝
  //(如果直接传入arr是不行的，因为大多数c++类的都是深拷贝，除了智能指针和原始指针)
  //int *arr_data = arr.data();
  // parallel_for<<<32, 128>>>(n, [=]__device__(int i){
  //   arr_data[i] = i;
  // });
  //上面这个方式麻烦点，还可以直接在[]里自定义捕获的表达式,这样可以使用同一变量名
  parallel_for<<<32, 128>>>(n, [arr = arr.data()]__device__(int i){
    arr[i] = i;
  });
  checkCudaErrors(cudaDeviceSynchronize());

  // for(int i = 0; i < n; i ++){
  //   printf("arr[%d]: %d\n", i, arr[i]);
  // }

  return 0;
}
