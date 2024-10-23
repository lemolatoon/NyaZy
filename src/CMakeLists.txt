# Locate all the .cpp files in the src directory
set(SRC_FILES
    main.cpp
    lexer.cpp
    ast.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# Create an executable for the main project from the source files
target_sources(nyacc PRIVATE ${SRC_FILES})

# Link with necessary libraries (e.g., LLVM, if needed)
# target_link_libraries(nyacc ${LLVM_LIBS})
target_link_libraries(nyacc
    PRIVATE
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
