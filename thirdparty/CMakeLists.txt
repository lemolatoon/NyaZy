cmake_minimum_required(VERSION 3.15)
project(nyazy-thirdparty)

include(ExternalProject)

# Specify the LLVM version and Git tag
set(LLVM_VERSION "llvmorg-19.1.2")
set(LLVM_REPO_URL "https://github.com/llvm/llvm-project.git")
set(LLVM_PROJECT_BUILD_DIR ${CMAKE_BINARY_DIR}/llvm-project/build)

# Set the directory where LLVM and MLIR are installed
set(LLVM_PROJECT_INSTALL_DIR ${CMAKE_BINARY_DIR}/llvm/install)

# https://stackoverflow.com/questions/45414507/pass-a-list-of-prefix-paths-to-externalproject-add-in-cmake-args
string(REPLACE ";" "|" CMAKE_PREFIX_PATH_ALT_SEP "${CMAKE_PREFIX_PATH}")

# Add LLVM as an external project
ExternalProject_Add(
    llvm_project
    PREFIX ${CMAKE_BINARY_DIR}/llvm
    GIT_REPOSITORY ${LLVM_REPO_URL}
    GIT_TAG ${LLVM_VERSION}
    SOURCE_SUBDIR llvm
    UPDATE_COMMAND ""
    LIST_SEPARATOR |
    CMAKE_ARGS
        -DLLVM_ENABLE_PROJECTS=clang|mlir
        -DLLVM_ENABLE_RUNTIMES=libcxx|libcxxabi|libunwind
        -DLLVM_BUILD_EXAMPLES=ON
        -DLLVM_BUILD_TOOLS=ON
        -DLLVM_TARGETS_TO_BUILD=Native
        -DCMAKE_BUILD_TYPE=Release
        -DLLVM_ENABLE_ASSERTIONS=ON
        -DCMAKE_C_COMPILER=clang
        -DCMAKE_CXX_COMPILER=clang++
        -DLLVM_ENABLE_LLD=ON
        -DLLVM_CCACHE_BUILD=ON
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/llvm/install
        -DLLVM_TOOL_CLANG_BUILD=ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build .
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
    USES_TERMINAL_BUILD TRUE
)