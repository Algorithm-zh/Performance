#include <cstdio>
#include <stdio.h>

//device定义设备函数，在gpu上执行，但是从gpu上调用,可以不是void
//inline声明内联函数,一般不需要用
__device__ __inline__ void say_Hello(){
  printf("Hello, world\n"); 
}

//这样定义的话cpu和gpu都能用
__host__ __device__ void say_hello_all(){
//一段代码会先送到cpu的编译器编译然后送到gpu编译器编译，而这个宏可以判断当前是cpu模式还是gpu模式
//cpu编译时会提取所有的host修饰的函数，gpu则是device的函数，所以这个函数会被编译两次的
#ifdef __CUDA_ARCH__
  printf("Hello, world gpu architecture %d\n", __CUDA_ARCH__);//52.版本号，编译时制定的版本，默认就是最老的
#else
  printf("Hello, world cpu\n"); 
#endif
}

//global定义核函数，在gpu上运行，不能有返回值,从cpu端调用,必须加<<<>>>
__global__ void kernal() { 
  say_hello_all();
}

//host声明只能在host使用，默认就是host
__host__ void say_hello_host(){
  printf("Hello, world host\n"); 
}

int main() {
  
  kernal<<<1, 1>>>();
  //cpu和gpu是异步的，所以需要让cpu等待gpu完成再返回
  cudaDeviceSynchronize();
  say_hello_all();
  return 0;
}
