cmake_minimum_required(VERSION 3.15)
project(nyazy LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

include(ExternalProject)

set(LLVM_DIR ${CMAKE_BINARY_DIR}/../thirdparty/build/llvm/install/lib/cmake/llvm)
set(MLIR_DIR ${CMAKE_BINARY_DIR}/../thirdparty/build/llvm/install/lib/cmake/mlir)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# mlir related settings -----
# ref: llvm-project/mlir/examples/standalone/CMakeLists.txt
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# include scripts
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
# ---------------------------

add_executable(nyacc)
target_compile_options(nyacc PRIVATE -Wall -Wextra -Werror -fno-rtti)

# Add subdirectories for src and test
add_subdirectory(src)
# add_subdirectory(test)
