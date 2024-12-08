#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

namespace nyacc {

// Helper function to get or create a format string global variable
mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::ModuleOp module,
                                    mlir::PatternRewriter &rewriter,
                                    llvm::StringRef formatStr);

} // namespace nyacc
