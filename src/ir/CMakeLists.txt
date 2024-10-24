target_sources(NyaZyDialect PRIVATE
    NyaZyDialect.cpp
)

target_link_libraries(NyaZyDialect
    MLIRIR
    MLIRSupport
    MLIRDialect
)