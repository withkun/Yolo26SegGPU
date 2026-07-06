cmake_minimum_required(VERSION 3.18)
project(kernel_segment LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)

################################################################################
set(CUDA_SDK_ROOT_DIR ${CUDA_PATH})
set(CMAKE_CUDA_COMPILER ${CUDA_PATH}/bin/nvcc.exe)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)


################################################################################
# NVCC和spdlog存在冲突无法正常编译, 拆分单独编译.
add_library(${PROJECT_NAME} STATIC
    segment.h
    segment.cu
    cuda_nms.cu
    cuda_gemv.cu
    cuda_upscale.cu
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_SDK_ROOT_DIR}/include
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ENABLE_FAST_MATH=1 DEBUG_MASK_KERNEL=1
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -O2 -use_fast_math --expt-relaxed-constexpr
)

target_link_options(${PROJECT_NAME} PRIVATE
    /DEBUG /NOLOGO
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${CUDA_SDK_ROOT_DIR}/lib/x64
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    cudart cublas
)