target_sources(NyaZyDialect PRIVATE
    NyaZyDialect.cpp
    NyaZyOpsEnums.cpp
    NyaZyOps.cpp
    lowerNyaZyPrintToPrintf.cpp
    lowerNyaZyToStd.cpp
    lowerStdToLLVM.cpp
    common.cpp
)

target_link_libraries(NyaZyDialect
    MLIRIR
    MLIRSupport
    MLIRDialect
)