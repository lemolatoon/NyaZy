target_sources(NyaZyDialect PRIVATE
    NyaZyDialect.cpp
    NyaZyOpsEnums.cpp
    NyaZyOps.cpp
    lowerToLLVM.cpp
)

target_link_libraries(NyaZyDialect
    MLIRIR
    MLIRSupport
    MLIRDialect
)