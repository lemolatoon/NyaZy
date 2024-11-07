#pragma once
#include "types.h"
#include <error.h>
#include <mlir/IR/Types.h>

namespace mlir {
template <class OpT> class OwningOpRef;
class ModuleOp;
class MLIRContext;
} // namespace mlir

namespace nyacc {
class ModuleAST;

mlir::Type asMLIRType(mlir::MLIRContext *ctx, Type type);
mlir::Type asMLIRType(mlir::MLIRContext *ctx, PrimitiveType type);

class MLIRGen {
public:
  static Result<mlir::OwningOpRef<mlir::ModuleOp>>
  gen(mlir::MLIRContext &context, const ModuleAST &moduleAst);
};

} // namespace nyacc
