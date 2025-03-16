// what should be portable to CUDA? 
// 1. scene, it will be accessed in CUDA kernel, so port everything in scene to device memory first, 
// like Triangle, Sphere, etc.
// 2. if a class has member pointer to another class, you need to make it portable and handle the alloc/free manually
#ifdef GPU_PATH_TRACER
#include <cuda.h>
#include <cuda_runtime.h>
#define CUDA_PORTABLE(CLASS_NAME) \
    void MallocCuda(CLASS_NAME*& device_ptr) const;\
    void FreeCuda() const;
#define FUNC_QUALIFIER __device__ __host__
#else
#define FUNC_QUALIFIER  
#define CUDA_PORTABLE(CLASS_NAME) 
#endif

// MallocCuda() construct a copy this object on device memory loacted at device_ptr, if device_ptr==nullptr,
// then it will allocate a new object on device memory and write its address to device_ptr.
// FreeCuda() should free the memory allocated by MallocCuda()