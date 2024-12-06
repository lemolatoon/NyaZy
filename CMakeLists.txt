cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0116 NEW)
# cmake_policy(SET CMP0135 NEW)
project(nyazy LANGUAGES CXX C)

set(LLVM_INSTALL_DIR ${CMAKE_BINARY_DIR}/../thirdparty/build/llvm/install)
set(LLVM_DIR ${LLVM_INSTALL_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_INSTALL_DIR}/lib/cmake/mlir)

set(tl-expected_INSTALL_DIR ${CMAKE_BINARY_DIR}/../thirdparty/build/tl-expected/install)
set(tl-expected_DIR ${tl-expected_INSTALL_DIR}/share/cmake/tl-expected)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


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

find_package(tl-expected REQUIRED CONFIG)
include_directories(${tl-expected_INSTALL_DIR}/include)


add_executable(nyac)
add_library(NYAC)
add_library(NyaZyDialect)
add_executable(simpleTest)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -fno-rtti -g")

include_directories(include)
add_subdirectory(include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_dependencies(NYAC MLIRNyaZyOpsIncGen)
add_dependencies(NYAC MLIRNyaZyDialectIncGen)

add_dependencies(nyac MLIRNyaZyOpsIncGen)
add_dependencies(nyac MLIRNyaZyDialectIncGen)

add_dependencies(simpleTest MLIRNyaZyOpsIncGen)
add_dependencies(simpleTest MLIRNyaZyDialectIncGen)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(src)
add_subdirectory(test)
