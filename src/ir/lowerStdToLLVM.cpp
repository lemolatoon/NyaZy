#include "ir/NyaZyDialect.h"
#include "ir/Pass.h"
#include "ir/NyaZyOps.h"
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {
// Lower standard dialects(arith, memref, scf, func) to LLVM dialect
class StdToLLVMPass : public mlir::PassWrapper<StdToLLVMPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StdToLLVMPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<nyacc::NyaZyDialect>();
  }

private:
  void runOnOperation() final;
};
} // namespace

void StdToLLVMPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  target.addIllegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect>();
  target.addLegalOp<nyacc::PrintOp>();

  mlir::RewritePatternSet patterns(&getContext());

  // * -> llvm
  mlir::LLVMTypeConverter typeConverter(&getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

  // scf -> llvm
  mlir::ConversionTarget scfTarget(getContext());
  mlir::LLVMTypeConverter scfTypeConverter(&getContext());
  scfTarget.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  scfTarget.addLegalOp<nyacc::PrintOp>();
  scfTarget.addIllegalDialect<mlir::scf::SCFDialect>();

  mlir::RewritePatternSet scfPatterns(&getContext());
  mlir::populateSCFToControlFlowConversionPatterns(scfPatterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(scfTypeConverter, scfPatterns);

  if (failed(
          applyFullConversion(getOperation(), scfTarget, std::move(scfPatterns)))) {
    signalPassFailure();
  }

  // memref -> llvm
  mlir::ConversionTarget target2(getContext());
  target2.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
  target2.addLegalOp<nyacc::PrintOp>();
  target2.addIllegalDialect<mlir::memref::MemRefDialect>();

  mlir::RewritePatternSet patterns2(&getContext());

  mlir::LLVMTypeConverter typeConverter2(&getContext());

  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter2, patterns2);

  if (failed(
          applyFullConversion(getOperation(), target2, std::move(patterns2)))) {
    signalPassFailure();
  }
}

// Lower standard dialects(arith, memref, scf, func) to LLVM dialect
std::unique_ptr<mlir::Pass> nyacc::createStdToLLVMPass() {
  return std::make_unique<StdToLLVMPass>();
}
