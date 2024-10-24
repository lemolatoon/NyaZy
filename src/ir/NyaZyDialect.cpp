#include "ir/NyaZyDialect.h"
#include "ir/NyaZyOps.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"

#include "NyaZyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "NyaZyOps.cpp.inc"

#pragma GCC diagnostic pop

namespace nyacc {

void NyaZyDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "NyaZyOps.cpp.inc"
    >();
}

} // nyacc