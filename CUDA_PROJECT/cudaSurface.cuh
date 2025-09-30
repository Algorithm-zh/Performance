#include "cudaArray.cuh"

template <class T>
struct CudaSurfaceAccessor{
  //要访问一个多维数组，必须先创建一个表面对象
  cudaSurfaceObject_t m_cuSuf;

  //cudaBoundaryModeTrapx,y,z越界就崩溃
  template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  __device__ __forceinline__ T read(int x, int y, int z) const {
    return surf3Dread<T>(m_cuSuf, x * sizeof(T), y, z, mode);
  }

  //cudaBoundaryModeTrap越界就把xyz钳制到原本的数组大小范围内
  template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  __device__ __forceinline__ void write(T val, int x, int y, int z) const {
    surf3Dwrite<T>(val, m_cuSuf, x * sizeof(T), y, z, mode);
  }

};

template <class T>
struct CudaSurface : CudaArray<T>{
  cudaSurfaceObject_t m_cuSuf{}; 
  explicit CudaSurface(uint3 const &_dim)
    : CudaArray<T>(_dim){
      cudaResourceDesc resDesc{};
      resDesc.resType = cudaResourceTypeArray;
      resDesc.res.array.array = CudaArray<T>::getArray();
      checkCudaErrors(cudaCreateSurfaceObject(&m_cuSuf, &resDesc));
    }

  //直接返回弱引用，浅拷贝
  cudaSurfaceObject_t getSurface() const{
    return m_cuSuf;
  }
  //访问者模式。返回一个包装过的对象
  CudaSurfaceAccessor<T> accessSurface() const{
    return {m_cuSuf};
  }
  ~CudaSurface(){
    checkCudaErrors(cudaDestroySurfaceObject(m_cuSuf));
  }
};
