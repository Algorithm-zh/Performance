project(cuda2 LANGUAGES CUDA CXX)
add_executable(cuda2 cudaSurface.cu)

#开启此选项即可让核函数使用lambda
target_compile_options(cuda2 PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_include_directories(cuda2 PUBLIC include)
