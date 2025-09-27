#include <cstdio>
#include <cuda_runtime.h>
#include "include/CudaAllocator.h"
#include "include/helper_cuda.h"
#include "include/tick.h"
#include <iostream>
#include <vector>

__global__ void parallel_sum(int *sum, int const *arr, int n){
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    //并行编程，所以必须用原子操作,但是很影响性能
    atomicAdd(&sum[0], arr[i]);
  }
}

//用原子操作很影响性能，所以可以先累加到局部变量，然后在一次性累加到全局的sum
__global__ void parallel_sum2(int *sum, int const *arr, int n){
  int local_sum = 0;
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    //并行编程，所以必须用原子操作,但是很影响性能
    local_sum += arr[i];
  }
  atomicAdd(&sum[0], local_sum);
}

//声明sum为比原数组小1024倍的数组
//然后在gpu上启动n / 1024个线程，每个负责数组中1024个数的求和，然后写入到sum的对应元素里
//这样每个线程都是写的不同地址就不需要原子操作了
//最后在把求出的数组在cpu上完成最终的求和
__global__ void parallel_sum3(int *sum, int const *arr, int n){
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n / 1024; i += blockDim.x * gridDim.x){
    int local_sum = 0;
    for(int j = i * 1024; j < i * 1024 + 1024; j ++){
      local_sum += arr[j];
    }
    sum[i] = local_sum;
  }
}

//上面的方法内部实际是串行的，因为数据是以来的，下一时刻的local_sum依赖上一时刻的local_sum
//而该方法将上面展开后，这样每个for循环内部都是没有数据依赖，从而是并行
__global__ void parallel_sum4(int *sum, int const *arr, int n){
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n / 1024; i += blockDim.x * gridDim.x){
    int local_sum[1024];
    for(int j = 0; j < 1024; j ++){
      local_sum[j] = arr[i * 1024 + j];
    }
    for(int j = 0; j < 512; j ++){
      local_sum[j] += local_sum[j + 512];
    }
    for(int j = 0; j < 256; j ++){
      local_sum[j] += local_sum[j + 256];
    }
    for(int j = 0; j < 128; j ++){
      local_sum[j] += local_sum[j + 128];
    }
    for(int j = 0; j < 64; j ++){
      local_sum[j] += local_sum[j + 64];
    }
    for(int j = 0; j < 32; j ++){
      local_sum[j] += local_sum[j + 32];
    }
    for(int j = 0; j < 16; j ++){
      local_sum[j] += local_sum[j + 16];
    }
    for(int j = 0; j < 8; j ++){
      local_sum[j] += local_sum[j + 8];
    }
    for(int j = 0; j < 4; j ++){
      local_sum[j] += local_sum[j + 4];
    }
    for(int j = 0; j < 2; j ++){
      local_sum[j] += local_sum[j + 2];
    }
    for(int j = 0; j < 1; j ++){
      local_sum[j] += local_sum[j + 1];
    }
    sum[i] = local_sum[0];
  }
}


//板块的共享内存
//将上面的无数据依赖的for变为真正的并行，就是把线程升级为板块
//把for升级为线程，然后把local_sum这个线程局部数组变为板块局部数组
//也就是使用__shared__
//但是因为线程束的概念，板块里1024个线程并不是同时执行的，而是32个32个的执行
//而他会一会执行这个线程，一会执行那个线程，会导致错误，所以需要进行同步
__global__ void parallel_sum5(int *sum, int const *arr, int n){

  //这里要禁止编译器优化局部数组，否则结果不对
  volatile __shared__ int local_sum[1024];
  int j = threadIdx.x;
  int i = blockIdx.x;
  local_sum[j] = arr[i * 1024 + j];
  __syncthreads();
  if(j < 512){
    local_sum[j] += local_sum[j + 512];
  }
  __syncthreads();
  if(j < 256){
    local_sum[j] += local_sum[j + 256];
  }
  __syncthreads();
  if(j < 128){
    local_sum[j] += local_sum[j + 128];
  }
  __syncthreads();
  if(j < 64){
    local_sum[j] += local_sum[j + 64];
  }
  __syncthreads();
  //剩下的32个线程是一个线程束的，他们是绝对并行的，所以不需要再进行同步
  if(j < 32){
    local_sum[j] += local_sum[j + 32];
  }
  if(j < 16){
    local_sum[j] += local_sum[j + 16];
  }
  if(j < 8){
    local_sum[j] += local_sum[j + 8];
  }
  if(j < 4){
    local_sum[j] += local_sum[j + 4];
  }
  if(j < 2){
    local_sum[j] += local_sum[j + 2];
  }
  if(j == 0){
    sum[i] = local_sum[0] + local_sum[1];
  }
}

//上面的在j < 32 的时候if判断时候会产生线程束分化，导致副作用，因此可以合并起来
__global__ void parallel_sum6(int *sum, int const *arr, int n){

  //这里要禁止编译器优化局部数组，否则结果不对
  volatile __shared__ int local_sum[1024];
  int j = threadIdx.x;
  int i = blockIdx.x;
  local_sum[j] = arr[i * 1024 + j];
  __syncthreads();
  if(j < 512){
    local_sum[j] += local_sum[j + 512];
  }
  __syncthreads();
  if(j < 256){
    local_sum[j] += local_sum[j + 256];
  }
  __syncthreads();
  if(j < 128){
    local_sum[j] += local_sum[j + 128];
  }
  __syncthreads();
  if(j < 64){
    local_sum[j] += local_sum[j + 64];
  }
  __syncthreads();
  //剩下的32个线程是一个线程束的，他们是绝对并行的，所以不需要再进行同步
  if(j < 32){
    local_sum[j] += local_sum[j + 32];
    local_sum[j] += local_sum[j + 16];
    local_sum[j] += local_sum[j + 8];
    local_sum[j] += local_sum[j + 4];
    local_sum[j] += local_sum[j + 2];
    if(j == 0){
      sum[i] = local_sum[0] + local_sum[1];
    }
  }
}

//使用模板函数包装加速数组求和
template <int blockSize, class T>
__global__ void parallel_sum_kernal(T *sum, T const *arr, int n){
  __shared__ volatile int local_sum[blockSize];
  int j = threadIdx.x;
  int i = blockIdx.x;
  T temp_sum = 0;
  //网格跨步循环,让一个板块处理多个板块
  for(int t = i * blockSize + j; t < n; t += blockSize * gridDim.x){
    temp_sum += arr[t];
  }
  local_sum[j] = temp_sum;
  __syncthreads();
  if constexpr (blockSize >= 1024){
    if(j < 512)
      local_sum[j] += local_sum[j + 512];
    __syncthreads();
  }
  if constexpr (blockSize >= 512){
    if(j < 256)
      local_sum[j] += local_sum[j + 256];
    __syncthreads();
  }
  if constexpr (blockSize >= 256){
    if(j < 128)
      local_sum[j] += local_sum[j + 128];
    __syncthreads();
  }
  if constexpr (blockSize >= 128){
    if(j < 64)
      local_sum[j] += local_sum[j + 64];
    __syncthreads();
  }
  if (j < 32){
    if constexpr (blockSize >= 64)
      local_sum[j] += local_sum[j + 32];
    if constexpr (blockSize >= 32)
      local_sum[j] += local_sum[j + 16];
    if constexpr (blockSize >= 16)
      local_sum[j] += local_sum[j + 8];
    if constexpr (blockSize >= 8)
      local_sum[j] += local_sum[j + 4];
    if constexpr (blockSize >= 4)
      local_sum[j] += local_sum[j + 2];
    if(j == 0){
      sum[i] = local_sum[0] + local_sum[1];
    }
  }
}

template <int reduceScale = 4096, int blockSize = 256, class T>
int parallel_sum(T const *arr, int n){
  std::vector<int, CudaAllocator<int>> sum(n / reduceScale);
  parallel_sum_kernal<blockSize><<<n / reduceScale, blockSize>>>(sum.data(), arr, n);
  checkCudaErrors(cudaDeviceSynchronize());
  T final_sum = 0;
  for(int i = 0; i < n / reduceScale; i ++){
    final_sum += sum[i];
  }
  return final_sum;
}

// 最终结果展示
// sum 1 result: 25172683
// parallel_sum3: 0.00125024s
// sum 3 result: 25172683
// parallel_sum4: 0.00228721s
// sum 4 result: 25172683
// parallel_sum5: 0.00159817s
// sum 5 result: 25172683
// parallel_sum6: 0.00140065s
// sum 6 result: 25172683
// parallel_sum_final: 0.00100452s
// sum final result: 25172683


int main(){

  int n = 1 << 24;
  std::vector<int, CudaAllocator<int>> arr(n);
  std::vector<int, CudaAllocator<int>> sum0(1);
  std::vector<int, CudaAllocator<int>> sum(1);
  std::vector<int, CudaAllocator<int>> sum3(n / 1024);
  std::vector<int, CudaAllocator<int>> sum4(n / 1024);
  std::vector<int, CudaAllocator<int>> sum5(n / 1024);
  std::vector<int, CudaAllocator<int>> sum6(n / 1024);


  for(int i = 0; i < n; i ++){
    arr[i] = std::rand() % 4;
  }

  //warmup
  parallel_sum<<<n / 128, 128>>>(sum0.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  printf("sum 0 result: %d\n", sum0[0]);


  TICK(parallel_sum)
  parallel_sum<<<n / 128, 128>>>(sum.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_sum)
  printf("sum 1 result: %d\n", sum[0]);


  TICK(parallel_sum3)
  parallel_sum3<<<n / 1024 / 128, 128>>>(sum3.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  int final_sum = 0;
  for(int i = 0; i < n / 1024; i ++){
    final_sum += sum3[i];
  }
  TOCK(parallel_sum3)
  printf("sum 3 result: %d\n", final_sum);


  TICK(parallel_sum4)
  parallel_sum4<<<n / 1024 / 128, 128>>>(sum4.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  final_sum = 0;
  for(int i = 0; i < n / 1024; i ++){
    final_sum += sum4[i];
  }
  TOCK(parallel_sum4)
  printf("sum 4 result: %d\n", final_sum);


  TICK(parallel_sum5)
  parallel_sum5<<<n / 1024, 1024>>>(sum5.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  final_sum = 0;
  for(int i = 0; i < n / 1024; i ++){
    final_sum += sum5[i];
  }
  TOCK(parallel_sum5)
  printf("sum 5 result: %d\n", final_sum);

  TICK(parallel_sum6)
  parallel_sum6<<<n / 1024, 1024>>>(sum6.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  final_sum = 0;
  for(int i = 0; i < n / 1024; i ++){
    final_sum += sum6[i];
  }
  TOCK(parallel_sum6)
  printf("sum 6 result: %d\n", final_sum);

  TICK(parallel_sum_final)
  final_sum = parallel_sum(arr.data(), n);
  TOCK(parallel_sum_final)
  printf("sum final result: %d\n", final_sum);
  

}
