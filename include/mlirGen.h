#pragma once

namespace mlir {
template <class OpT> class OwningOpRef;
class ModuleOp;
class MLIRContext;
} // namespace mlir

namespace nyacc {
class ModuleAST;

class MLIRGen {
public:
  static mlir::OwningOpRef<mlir::ModuleOp> gen(mlir::MLIRContext &context,
                                               const ModuleAST &moduleAst);
};

} // namespace nyacc