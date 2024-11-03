#include "ir/NyaZyDialect.h"
#include "ir/NyaZyOps.h"
#include "ir/Pass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>

namespace {

// mlir::FlatSymbolRefAttr getOrInsertExit(mlir::ModuleOp module) {
//     auto* context = module.getContext();
//     const char *exitSymbol = "exit";
//
//     if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(exitSymbol)) {
//         return mlir::SymbolRefAttr::get(context, exitSymbol);
//     }
//
//     auto llvmExitType =
//     mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(context),
//     {mlir::IntegerType::get(context, 64)});
//
//     mlir::PatternRewriter rewriter{context};
//     mlir::PatternRewriter::InsertionGuard guard(rewriter);
//     rewriter.setInsertionPointToStart(module.getBody());
//     rewriter.create<mlir::LLVM::LLVMFuncOp>(
//         module.getLoc(), exitSymbol, llvmExitType);
//
//     return mlir::SymbolRefAttr::get(context, exitSymbol);
// }

class ConstantOpLowering : public mlir::OpConversionPattern<nyacc::ConstantOp> {
public:
  explicit ConstantOpLowering(mlir::MLIRContext *context)
      : OpConversionPattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(nyacc::ConstantOp op, OpAdaptor adaptor [[maybe_unused]],
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto constantOp = mlir::cast<nyacc::ConstantOp>(op);
    rewriter.replaceOp(op, rewriter.create<mlir::arith::ConstantOp>(
                               op->getLoc(), constantOp.getValue()));

    return mlir::success();
  }
};

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public mlir::OpConversionPattern<BinaryOp> {
  BinaryOpLowering(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<BinaryOp>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor [[maybe_unused]],
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto binOp = mlir::cast<BinaryOp>(op);
    rewriter.replaceOp(op, rewriter.create<LoweredBinaryOp>(
                               op->getLoc(), binOp.getLhs(), binOp.getRhs()));

    return mlir::success();
  }
};
using AddOpLowering = BinaryOpLowering<nyacc::AddOp, mlir::arith::AddIOp>;
using SubOpLowering = BinaryOpLowering<nyacc::SubOp, mlir::arith::SubIOp>;
using MulOpLowering = BinaryOpLowering<nyacc::MulOp, mlir::arith::MulIOp>;
using DivOpLowering = BinaryOpLowering<nyacc::DivOp, mlir::arith::DivSIOp>;

struct PosOpLowering : public mlir::OpConversionPattern<nyacc::PosOp> {
  PosOpLowering(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<nyacc::PosOp>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(nyacc::PosOp op, nyacc::PosOp::Adaptor adaptor [[maybe_unused]],
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto unaryOp = mlir::cast<nyacc::PosOp>(op);
    rewriter.replaceOp(op, unaryOp.getOperand());

    return mlir::success();
  }
};

struct NegOpLowering : public mlir::OpConversionPattern<nyacc::NegOp> {
  NegOpLowering(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<nyacc::NegOp>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(nyacc::NegOp op, nyacc::NegOp::Adaptor adaptor [[maybe_unused]],
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto unaryOp = mlir::cast<nyacc::NegOp>(op);
    auto cst0 = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    rewriter.replaceOp(op, rewriter.create<mlir::arith::SubIOp>(
                               op->getLoc(), cst0, unaryOp.getOperand()));

    return mlir::success();
  }
};

struct CmpOpLowering : public mlir::OpConversionPattern<nyacc::CmpOp> {
  CmpOpLowering(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<nyacc::CmpOp>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(nyacc::CmpOp op, nyacc::CmpOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
		auto pred = adaptor.getPredicate();
		mlir::arith::CmpIPredicate arithPred;
		// TODO: assume signed integer here.
		switch (pred) {
		case nyacc::CmpPredicate::eq:
			arithPred =  mlir::arith::CmpIPredicate::eq;
			break;
		case nyacc::CmpPredicate::ne:
			arithPred =  mlir::arith::CmpIPredicate::ne;
			break;
		case nyacc::CmpPredicate::lt:
			arithPred =  mlir::arith::CmpIPredicate::slt;
			break;
		case nyacc::CmpPredicate::le:
			arithPred =  mlir::arith::CmpIPredicate::sle;
			break;
		case nyacc::CmpPredicate::gt:
			arithPred =  mlir::arith::CmpIPredicate::sgt;
			break;
		case nyacc::CmpPredicate::ge:
			arithPred =  mlir::arith::CmpIPredicate::sge;
			break;
		}
		// auto arithPredAttr = mlir::arith::invertPredicate(arithPred);
		auto loc = op->getLoc();
		auto arithCmpOp = rewriter.create<mlir::arith::CmpIOp>(loc, arithPred, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, arithCmpOp);

    return mlir::success();
  }
};

class ReturnOpLowering : public mlir::OpRewritePattern<nyacc::ReturnOp> {
public:
  explicit ReturnOpLowering(mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(nyacc::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, op->getOperands());

    return mlir::success();
  }
};

struct FuncOpLowering : public mlir::OpConversionPattern<nyacc::FuncOp> {
  using OpConversionPattern<nyacc::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(nyacc::FuncOp op, OpAdaptor adaptor [[maybe_unused]],
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main")
      return mlir::failure();

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](mlir::Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    auto mainFuncType = mlir::FunctionType::get(rewriter.getContext(), {},
                                                {rewriter.getI64Type()});

    // Create a new non-toy function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    mainFuncType);
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

class NyaZyToLLVMPass
    : public mlir::PassWrapper<NyaZyToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NyaZyToLLVMPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<nyacc::NyaZyDialect>();
  }

private:
  void runOnOperation() final;
};

} // namespace

void NyaZyToLLVMPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<nyacc::NyaZyDialect>();

  mlir::RewritePatternSet patterns(&getContext());
  // nyazy -> arith + func
  patterns.add<ConstantOpLowering, FuncOpLowering, ReturnOpLowering,
               AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering, PosOpLowering, NegOpLowering, CmpOpLowering>(
      &getContext());

  // * -> llvm
  mlir::LLVMTypeConverter typeConverter(&getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> nyacc::createNyaZyToLLVMPass() {
  return std::make_unique<NyaZyToLLVMPass>();
}
