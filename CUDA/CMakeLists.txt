cmake_minimum_required(VERSION 3.18)
project(cuda LANGUAGES CUDA CXX)
add_executable(cuda sum.cu)

#可以通过这个设置使用版本 默认最老的,为了兼容,而我们可以指定多个这样他可以自己检测自动选择(但是编译会变慢，最好不要这样)
#set(CMAKE_CUDA_ARCHITECTURES 52;70;75;86)

#启用这个可以分离声明和定义(对所有程序)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

#指定这个选项就可以使用constexpr,从而让两边都可以调用函数
#target_compile_options(cuda PUBLIC )

#开启此选项即可让核函数使用lambda
target_compile_options(cuda PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_include_directories(cuda PUBLIC include)
