#include "include/CudaAllocator.h"
#include "include/helper_cuda.h"
#include "include/tick.h"
#include <cstdio>
#include <iostream>
#include <vector>

//矩阵转置
template <class T>
__global__ void parallel_transpose(T *out, T const *in, int nx, int ny){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || y >= ny)return;
  out[y * nx + x] = in[x * nx + y];//in读取是跨步的
}

//全局内存跨步比较慢，可以先读到共享内存里，速度快
template <int blockSize, class T>
__global__ void parallel_transpose2(T *out, T const *in, int nx, int ny){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || y >= ny)return;
  __shared__ T tmp[blockSize * blockSize];
  int rx = blockIdx.y * blockSize + threadIdx.x;
  int ry = blockIdx.x * blockSize + threadIdx.y;
  tmp[threadIdx.y * blockSize + threadIdx.x] = in[ry * nx + rx];
  __syncthreads();
  out[y * nx + x] = tmp[threadIdx.x * blockSize + threadIdx.y];
}

//因为bank的存在，所以将其错位,这样线程0访问arr[0]的位于bank0,线程1访问的arr[33]位于bank1
//正好变成了一个线程访问一个bank,没有冲突，不用排队
template <int blockSize, class T>
__global__ void parallel_transpose3(T *out, T const *in, int nx, int ny){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || y >= ny)return;
  __shared__ T tmp[(blockSize + 1) * blockSize];
  int rx = blockIdx.y * blockSize + threadIdx.x;
  int ry = blockIdx.x * blockSize + threadIdx.y;
  tmp[threadIdx.y * (blockSize + 1) + threadIdx.x] = in[ry * nx + rx];
  __syncthreads();
  out[y * nx + x] = tmp[threadIdx.x * (blockSize + 1) + threadIdx.y];
}

int main(){
  int nx = 1<<12, ny = 1<<12;
  std::vector<int ,CudaAllocator<int>> in(nx * ny);
  std::vector<int ,CudaAllocator<int>> out(nx * ny);
  std::vector<int ,CudaAllocator<int>> out2(nx * ny);
  std::vector<int ,CudaAllocator<int>> out3(nx * ny);

  for(int i = 0; i < nx * ny; i ++){
    in[i] = i;
  }

  TICK(parallel_transpose)
  parallel_transpose<<<dim3(nx / 32, ny / 32, 1),
                      dim3(32, 32, 1)>>>
                      (out.data(), in.data(), nx, ny);
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_transpose)

  for(int y = 0; y < ny; y ++){
    for(int x = 0; x < nx; x ++){
      if(out[y * nx + x] != in[x * nx + y]){
        printf("Wrong At x = %d, y = %d: %d != %d\n", x, y,
              out[y * nx + x], in[x * nx + y]);
        return -1;
      }
    }
  }

  TICK(parallel_transpose2)
  parallel_transpose2<32><<<dim3(nx / 32, ny / 32, 1),
                      dim3(32, 32, 1)>>>
                      (out2.data(), in.data(), nx, ny);
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_transpose2)

  TICK(parallel_transpose3)
  parallel_transpose3<32><<<dim3(nx / 32, ny / 32, 1),
                      dim3(32, 32, 1)>>>
                      (out3.data(), in.data(), nx, ny);
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_transpose3)



  printf("Correct!\n");
  return 0;
}
