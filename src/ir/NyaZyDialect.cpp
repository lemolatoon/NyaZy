#include "ir/NyaZyDialect.h"
#include "ir/NyaZyOps.h"
#include <mlir/IR/OpDefinition.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"

#include "ir/NyaZyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "ir/NyaZyOps.cpp.inc"

#pragma GCC diagnostic pop

namespace nyacc {

void NyaZyDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "ir/NyaZyOps.cpp.inc"
    >();
}

} // nyacc