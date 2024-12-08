// Helper function to get or create a format string global variable
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

namespace nyacc {
mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::ModuleOp module,
                                    mlir::PatternRewriter &rewriter,
                                    llvm::StringRef formatStr) {
  mlir::LLVM::GlobalOp globalStr;

  // Use the format string as the global variable name
  std::string globalName = "global_str_" + std::to_string(std::hash<std::string>{}(formatStr.str()));

  if (!(globalStr = module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName))) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    mlir::Type i8Type = mlir::IntegerType::get(rewriter.getContext(), 8);
    mlir::Type strType = mlir::LLVM::LLVMArrayType::get(i8Type, formatStr.size() + 1);

    globalStr = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, strType, /*isConstant=*/true, mlir::LLVM::Linkage::Internal,
        globalName, rewriter.getStringAttr(formatStr.str() + '\0'));
  }

  // Get a pointer to the first character in the global string
  mlir::Type i8PtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

  mlir::Value globalPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, globalStr);
  mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
  mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(
      loc, i8PtrType, globalStr.getType(), globalPtr, mlir::ValueRange{zero, zero});

  return ptr;
}

} // namespace nyacc