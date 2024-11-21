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
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
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
    auto u1Type = rewriter.getIntegerType(1, false);
    auto unsignedCmpResult = rewriter.create<mlir::arith::BitcastOp>(loc, u1Type, arithCmpOp);
    rewriter.replaceOp(op, unsignedCmpResult);

    return mlir::success();
  }
};

struct CastOpLowering : public mlir::OpConversionPattern<nyacc::CastOp> {
  CastOpLowering(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<nyacc::CastOp>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(nyacc::CastOp op, nyacc::CastOp::Adaptor adaptor [[maybe_unused]],
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto fromT = op.getIn().getType();
    auto toT = op.getType();
		auto loc = op->getLoc();
    if (fromT == toT) {
      rewriter.replaceOp(op, op.getIn());
      return mlir::success();
    }

    const auto int2fp = [&](mlir::Value from, mlir::FloatType floatType) -> mlir::Value {
      assert(from.getType().isInteger());
      if (from.getType().isUnsignedInteger()) {
        return rewriter.create<mlir::arith::UIToFPOp>(loc, floatType, from);
      } else {
        return rewriter.create<mlir::arith::SIToFPOp>(loc, floatType, from);
      }
    };
    
    const auto fp2int = [&](mlir::Value from, mlir::IntegerType intType) -> mlir::Value {
      assert(from.getType().isIntOrFloat() && !from.getType().isInteger());
      if (from.getType().isUnsignedInteger()) {
        return rewriter.create<mlir::arith::FPToUIOp>(loc, intType, from);
      } else {
        return rewriter.create<mlir::arith::FPToSIOp>(loc, intType, from);
      }
    };

    const auto int2intExt = [&](mlir::Value from, mlir::IntegerType intType) -> mlir::Value {
      assert(from.getType().getIntOrFloatBitWidth() < intType.getIntOrFloatBitWidth());
      if (from.getType().isUnsignedInteger()) {
        return rewriter.create<mlir::arith::ExtUIOp>(loc, intType, from);
      } else {
        return rewriter.create<mlir::arith::ExtSIOp>(loc, intType, from);
      }
    };

    const auto int2intTrunc = [&](mlir::Value from, mlir::IntegerType intType) -> mlir::Value {
      assert(from.getType().getIntOrFloatBitWidth() > intType.getIntOrFloatBitWidth());
      return rewriter.create<mlir::arith::TruncIOp>(loc, intType, from);
    };

    auto value = op.getIn();
    if (toT.isInteger()) {
      const mlir::IntegerType intToT = llvm::cast<mlir::IntegerType>(toT);
      if (!fromT.isInteger()) {
        value = fp2int(value, intToT);
      } else {
        // int to int
        if (fromT.getIntOrFloatBitWidth() > toT.getIntOrFloatBitWidth()) {
          // truncate
          value = int2intTrunc(value, intToT);
        } else {
          // extend
          value = int2intExt(value, intToT);
        }
      }
    } else {
      value = int2fp(value, llvm::cast<mlir::FloatType>(toT));
    }

    rewriter.replaceOp(op, value);

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

// Lowering for nyazy.alloca -> memref.alloca
struct AllocaOpLowering : public mlir::OpConversionPattern<nyacc::AllocaOp> {
  using OpConversionPattern<nyacc::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      nyacc::AllocaOp op, OpAdaptor adaptor [[maybe_unused]],
      mlir::ConversionPatternRewriter &rewriter) const final {
    if (!llvm::isa<mlir::MemRefType>(op.getType())) {
      return mlir::failure();
    }
    auto memrefType = llvm::cast<mlir::MemRefType>(op.getType());
    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(op, memrefType);
    return mlir::success();
  }
};

// Lowering for nyazy.load -> memref.load
struct LoadOpLowering : public mlir::OpConversionPattern<nyacc::LoadOp> {
  using OpConversionPattern<nyacc::LoadOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      nyacc::LoadOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    auto op2 = rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(
        op, adaptor.getMemref());
    llvm::errs() << "LoadOpLowering: \n";
    op2.print(llvm::errs());
    return mlir::success();
  }
};

// Lowering for nyazy.store -> memref.store
struct StoreOpLowering : public mlir::OpConversionPattern<nyacc::StoreOp> {
  using OpConversionPattern<nyacc::StoreOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      nyacc::StoreOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref());
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
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  target.addIllegalDialect<nyacc::NyaZyDialect>();

  mlir::RewritePatternSet patterns(&getContext());
  // nyazy -> arith + func
  patterns.add<ConstantOpLowering, FuncOpLowering, ReturnOpLowering,
               AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering, 
               PosOpLowering, NegOpLowering, CmpOpLowering, CastOpLowering,
               AllocaOpLowering, LoadOpLowering, StoreOpLowering>(
      &getContext());

  // * -> llvm
  mlir::LLVMTypeConverter typeConverter(&getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

  // memref -> llvm
  mlir::ConversionTarget target2(getContext());
  target2.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
  target2.addIllegalDialect<mlir::memref::MemRefDialect>();

  mlir::RewritePatternSet patterns2(&getContext());

  mlir::LLVMTypeConverter typeConverter2(&getContext());

  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter2, patterns2);

  if (failed(
          applyFullConversion(getOperation(), target2, std::move(patterns2)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> nyacc::createNyaZyToLLVMPass() {
  return std::make_unique<NyaZyToLLVMPass>();
}
