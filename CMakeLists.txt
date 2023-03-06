cmake_minimum_required(VERSION 3.18...3.26)

project(jaxfftMp LANGUAGES CXX CUDA)

set(ENV{CUDA_TOOLKIT_ROOT_DIR} /usr/local/cuda)
link_directories(/usr/local/cuda/lib64)

list(APPEND CMAKE_PREFIX_PATH "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cmake")
find_package(NVHPC REQUIRED COMPONENTS MATH MPI)

add_subdirectory(pybind11)

# Adding a bunch of inputs
include_directories("${NVHPC_ROOT_DIR}/math_libs/include" "${NVHPC_ROOT_DIR}/math_libs/include/cufftmp" "${NVHPC_ROOT_DIR}/comm_libs/nccl/include")
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
link_directories("${NVHPC_ROOT_DIR}/math_libs/lib64")

pybind11_add_module(_cufftMp lib/cufftMp_wrapper.cc)
target_link_libraries(_cufftMp PRIVATE MPI::MPI_CXX NVHPC::MPI NVHPC::CUFFT cudart cufftMp)
