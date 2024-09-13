cmake_minimum_required(VERSION 3.0.2)
project(cf_cpu)
set( EXECUTABLE_NAME main.out)

set(CMAKE_C_COMPILER "/usr/lib/nvidia-cuda-toolkit/bin/gcc" CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER "/usr/lib/nvidia-cuda-toolkit/bin/g++" CACHE PATH "" FORCE)
# SET(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr" CACHE STRING "nvcc flags" FORCE)

MESSAGE( STATUS ">> CMAKE_C_COMPILER:        " ${CMAKE_C_COMPILER} )
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER:      " ${CMAKE_CXX_COMPILER} )


set(CUDA_DIR "/usr/lib/nvidia-cuda-toolkit")
find_package(CUDA REQUIRED)
add_definitions(-DUSE_CUDA)


set(HEADER_FILES
    # include/dot.hpp
)

set(SOURCE_FILES
    src/main.cpp
)


include_directories( PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)
cuda_add_executable( ${EXECUTABLE_NAME} ${SOURCE_FILES} )

target_link_libraries( ${EXECUTABLE_NAME} ${CUDA_LIBRARIES})

target_include_directories( ${EXECUTABLE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories( ${EXECUTABLE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty)