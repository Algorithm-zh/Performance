#include <cstdio>
#include <cuda_runtime.h>

//通过atomicCAS可以实现任意cuda没有提供的原子读-修改-写回指令
//但是很影响性能
__device__ __inline__ int my_atomic_mul(int *dst, int src){
  int old = *dst, expect;
  do{
    expect = old;
    //如果dst和expect不同就不写入，也就是当你要写入的值已经被改了，就不写了，就是原子操作
    old = atomicCAS(dst, expect, expect * src);
  } while(expect != old);
  return old;
}

__global__ void parallel_mul(int *sum, int const *arr, int n){
  int local_sum = 1;
  for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    local_sum *= arr[i]; 
  }
  my_atomic_mul(&sum[0], local_sum);
}
