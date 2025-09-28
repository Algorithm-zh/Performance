cmake_minimum_required(VERSION 3.18)
project(Performance)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

add_subdirectory(RALL_STL_PTR)
add_subdirectory(TEMPLATE)
add_subdirectory(THREAD)
add_subdirectory(TBB)
add_subdirectory(CUDA)
