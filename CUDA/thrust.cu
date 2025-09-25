#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "include/helper_cuda.h"

template <class Func>
__global__ void parallel_for(int n, Func func){
  //每个线程执行4个循环
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    func(i);
  }
}

int main(){
  int n = 65536;//2^16
  float a = 3.14f;
  //1.universal_vector会在统一内存上分配
  thrust::universal_vector<float> x(n);
  thrust::universal_vector<float> y(n);
  for(int i = 0; i < n; i ++){
    x[i] = std::rand() * (1.f / RAND_MAX);
    y[i] = std::rand() * (1.f / RAND_MAX);
  }
//7 7
  // parallel_for<<<n / 512, 128>>>(n, [a, x = x.data(), y = y.data()]__device__(int i){
  //   x[i] = a * x[i] + y[i];
  // });
  // checkCudaErrors(cudaDeviceSynchronize());

  for(int i = 0; i < n; i ++){
    printf("x[%d] = %f\n", i, x[i]);
  }


  //2.也可以使用分开的
  thrust::host_vector<float> x_host(n);
  thrust::host_vector<float> y_host(n);

  auto float_rand = []{
    return std::rand() * (1.f / RAND_MAX); 
  };

  //thrust提供了很多类似于标准库里的模板函数
  //generate(b, e, f) 批量调用f写入到[b, e)区间
  thrust::generate(x_host.begin(), x_host.end(), float_rand);
  thrust::generate(y_host.begin(), y_host.end(), float_rand);

  //可以直接使用=在他们之间拷贝数据，底层封装了cudaMemecpy
  thrust::device_vector<float> x_dev = x_host;
  thrust::device_vector<float> y_dev = y_host;
  parallel_for<<<n / 512, 128>>>(n, [a, x_dev = x_dev.data(), y_dev = y_dev.data()]__device__(int i){
    x_dev[i] = a * x_dev[i] + y_dev[i];
  });

  //thrust函数可以根据容器类型自动决定在cpu还是gpu上执行
  thrust::for_each(x_dev.begin(), x_dev.end(), []__device__(float &x){
    x += 100.f;
  });

  //make_zip_iterator可以将多个迭代器合并起来，然后在函数体里通过const const &tup捕获
  thrust::for_each(
    thrust::make_zip_iterator(x_dev.begin(), y_dev.cbegin()),
    thrust::make_zip_iterator(x_dev.end(), y_dev.cend()),
    [a] __device__(auto const &tup){
    auto &x = thrust::get<0>(tup);
    auto const &y = thrust::get<1>(tup);
    x = a * x + y;
  });

  x_host = x_dev;

  for(int i = 0; i < n; i ++){
    printf("x_host[%d] = %f\n", i, x_host[i]);
  }

  return 0;
}
