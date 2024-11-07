add_subdirectory(ir)

# Locate all the .cpp files in the src directory
set(SRC_FILES
    lexer.cpp
    ast.cpp
    parser.cpp
    mlirGen.cpp
    error.cpp
    types.cpp
    scope.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

message(STATUS "nyazy dialect sources: ${nyazy_dialect_sources}")
# Create an executable for the main project from the source files
target_sources(libNYACC PRIVATE ${SRC_FILES} ${nyazy_dialect_sources})

# Link with necessary libraries (e.g., LLVM, if needed)
# target_link_libraries(nyacc ${LLVM_LIBS})
target_link_libraries(libNYACC
    PRIVATE
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
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
)

mlir_check_link_libraries(libNYACC)

target_sources(nyacc PRIVATE main.cpp)
target_link_libraries(nyacc libNYACC)
