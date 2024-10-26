# test/CMakeLists.txt
enable_testing()  # CTest を有効にする

# テストターゲットの追加
set(SRC_FILES simpleTest.cpp)
target_sources(simpleTest PRIVATE ${SRC_FILES})

# テストを登録
add_test(NAME SimpleTest COMMAND simpleTest)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    list(APPEND LLVM_TARGET_COMPONENTS
        AArch64
        AArch64AsmParser
        AArch64CodeGen
        AArch64Desc
        AArch64Disassembler
        AArch64Info
        AArch64Utils
        ExecutionEngine
        OrcJIT
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
    list(APPEND LLVM_TARGET_COMPONENTS
        X86
        X86AsmParser
        X86CodeGen
        X86Desc
        X86Disassembler
        X86Info
        ExecutionEngine
        OrcJIT
    )
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Map components to library names
llvm_map_components_to_libnames(LLVM_TARGET_LIBS ${LLVM_TARGET_COMPONENTS})

message(STATUS "LLVM_TARGET_LIBS: ${LLVM_TARGET_LIBS}")

target_link_libraries(simpleTest
    PRIVATE
    libNYACC
    NyaZyDialect
    ${dialect_libs}
    ${extension_libs}
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRDialect 
    MLIRTranslateLib
    MLIRSupport
    MLIRTransforms

    ${LLVM_TARGET_LIBS}

    gtest gtest_main
)
