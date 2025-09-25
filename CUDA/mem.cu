#include <cstdio>
#include <cuda_runtime.h>
///opt/cuda/samples/common/inc/helper_cuda.h定义了很多函数和宏可以帮我快速检查错误
//找不到可以网上搜一下
//把helper_cuda.h和helper_string.h放到根目录里
#include "include/helper_cuda.h"

__global__ void kernal(int *pret){
  *pret = 42;
}

int main(){
  //int ret = 0;
  //无论在栈上还是堆上都会报错，因为gpu是独立的显存，不能访问cpu的内存
  //kernal<<<1,1>>>(&ret);
  int *pret;
  //可以用cudaMalloc分配显存,同样的cpu也不能访问gpu内存
  checkCudaErrors(cudaMalloc(&pret, sizeof(int)));
  kernal<<<1, 1>>>(pret);
  //有了cudaMemcpy不需要显示同步
  //checkCudaErrors(cudaDeviceSynchronize());

  int ret;
  //通过cudaMemcpy可以在cpu和gpu之间拷贝数据
  //并且cudaMemcpy会自动进行同步操作，所以不需要进行cudaDeviceSynchronize
  checkCudaErrors(cudaMemcpy(&ret, pret, sizeof(int), cudaMemcpyDeviceToHost));
  printf("result: %d\n", ret);

  cudaFree(pret);
  return 0;
}
