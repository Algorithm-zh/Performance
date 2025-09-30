#pragma once
#include "include/helper_cuda.h"
#include <cstdio>
#include <iostream>

//cuda多维数组封装
struct DisableCopy{
  DisableCopy() = default;
  DisableCopy(DisableCopy const &) = delete;
  DisableCopy &operator=(DisableCopy const &) = delete;
};

template <class T>
struct CudaArray : DisableCopy{
  //gpu的多维数组有特殊的数据排布来保障访存的高效
  cudaArray *m_cuArray{};
  uint3 m_dim{};

  explicit CudaArray(uint3 const &_dim)
    : m_dim(_dim){
    cudaExtent extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    //分配一个三位数组，各维度大小通过cudaExtent指定
    checkCudaErrors(cudaMalloc3DArray(&m_cuArray, &channelDesc, extent, cudaArraySurfaceLoadStore));
  }

  void copyIn(T const *_data){
    //通过结构体来传递参数
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcPtr = make_cudaPitchedPtr((void *)_data, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
    copy3DParams.dstArray = m_cuArray;
    copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
    copy3DParams.kind = cudaMemcpyHostToDevice;
    //使用cudaMemcpy3D在cpu和gpu的三维数组之间拷贝数据
    checkCudaErrors(cudaMemcpy3D(&copy3DParams));
  }

  void copyOut(T const *_data){
    //通过结构体来传递参数
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcArray = m_cuArray;
    copy3DParams.dstPtr = make_cudaPitchedPtr((void *)_data, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
    copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
    copy3DParams.kind = cudaMemcpyDeviceToHost;
    //使用cudaMemcpy3D在cpu和gpu的三维数组之间拷贝数据
    checkCudaErrors(cudaMemcpy3D(&copy3DParams));
  }

  cudaArray *getArray() const{
    return m_cuArray;
  }

  ~CudaArray(){
    checkCudaErrors(cudaFreeArray(m_cuArray));
  }
};

