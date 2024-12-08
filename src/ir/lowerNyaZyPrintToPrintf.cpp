#include "ir/common.h"
#include "ir/Pass.h"
#include "ir/NyaZyDialect.h"
#include "ir/NyaZyOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/IR/BuiltinDialect.h>

namespace {


// Helper function to get or insert the 'printf' function declaration
mlir::LLVM::LLVMFuncOp getOrInsertPrintf(mlir::ModuleOp module, mlir::PatternRewriter &rewriter) {
  auto printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
  if (printfFunc) {
    return printfFunc;
  }

  // Create 'printf' function declaration
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  mlir::Type i8PtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

  auto printfType = mlir::LLVM::LLVMFunctionType::get(
      mlir::IntegerType::get(rewriter.getContext(), 32),
      {i8PtrType}, /*isVarArg=*/true);

  printfFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "printf", printfType);

  return printfFunc;
}


struct PrintIntOpLowering : public mlir::OpConversionPattern<nyacc::PrintOp> {
  using mlir::OpConversionPattern<nyacc::PrintOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      nyacc::PrintOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // Check if the operand is an integer type
    auto operandType = adaptor.getOperand().getType();
    if (!llvm::isa<mlir::IntegerType>(operandType)) {
      return mlir::failure();
    }

    mlir::Location loc = op.getLoc();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();

    // Get or create the printf function declaration
    auto printfFunc = getOrInsertPrintf(module, rewriter);

    // Get or create the format string for integers
    auto formatStrPtr = nyacc::getOrCreateGlobalString(loc, module, rewriter, "%ld\n");

    // Ensure the operand is of LLVM integer type (i64)
    mlir::Value value = adaptor.getOperand();

    // Create the printf call
    auto callop = rewriter.create<mlir::LLVM::CallOp>(
      loc,
        mlir::TypeRange{mlir::IntegerType::get(rewriter.getContext(), 32)},
        mlir::SymbolRefAttr::get(printfFunc),
        mlir::ValueRange{formatStrPtr, value});
    
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Type printfType = mlir::LLVM::LLVMFunctionType::get(
        mlir::IntegerType::get(rewriter.getContext(), 32),
        {ptrType}, /*isVarArg=*/true);
    callop->setAttr("var_callee_type", mlir::TypeAttr::get(printfType));

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

struct PrintStringOpLowering : public mlir::OpConversionPattern<nyacc::PrintOp> {
  using mlir::OpConversionPattern<nyacc::PrintOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      nyacc::PrintOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // Check if the operand is an LLVM pointer to i8 (string)
    auto operandType = adaptor.getOperand().getType();
    auto llvmPtrType = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(operandType);
    if (!llvmPtrType) {
      return mlir::failure();
    }

    mlir::Location loc = op.getLoc();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();

    // Get or create the printf function declaration
    auto printfFunc = getOrInsertPrintf(module, rewriter);

    // Get or create the format string for strings
    auto formatStrPtr = nyacc::getOrCreateGlobalString(loc, module, rewriter, "%s\n");

    // Create the printf call
    auto callop = rewriter.create<mlir::LLVM::CallOp>(
        loc,
        mlir::TypeRange{mlir::IntegerType::get(rewriter.getContext(), 32)},
        mlir::SymbolRefAttr::get(printfFunc),
        mlir::ValueRange{formatStrPtr, adaptor.getOperand()}
    );
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Type printfType = mlir::LLVM::LLVMFunctionType::get(
        mlir::IntegerType::get(rewriter.getContext(), 32),
        {ptrType}, /*isVarArg=*/true);
    callop->setAttr("var_callee_type", mlir::TypeAttr::get(printfType));

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

// Lower standard dialects(arith, memref, scf, func) to LLVM dialect
class StdToLLVMPass : public mlir::PassWrapper<StdToLLVMPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StdToLLVMPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<nyacc::NyaZyDialect, mlir::arith::ArithDialect, mlir::memref::MemRefDialect, mlir::scf::SCFDialect, mlir::func::FuncDialect>();
  }

private:
  void runOnOperation() final;
};
} // namespace

void StdToLLVMPass::runOnOperation() {
  // Lowering for nyazy.print -> LLVM printf
  mlir::ConversionTarget printTarget(getContext());
  printTarget.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
  printTarget.addIllegalOp<nyacc::PrintOp>();

  mlir::RewritePatternSet printPatterns(&getContext());
  printPatterns.add<PrintIntOpLowering, PrintStringOpLowering>(&getContext());

  if (failed(
          applyFullConversion(getOperation(), printTarget, std::move(printPatterns)))) {
    signalPassFailure();
  }

}

// Lower `nyazy.print` to LLVM printf
std::unique_ptr<mlir::Pass> nyacc::createNyaZyPrintToLLVMPass() {
  return std::make_unique<StdToLLVMPass>();
}
