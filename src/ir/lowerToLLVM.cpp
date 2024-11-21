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
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

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
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(
        op, adaptor.getMemref());
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

struct WhileOpLowering : public mlir::OpConversionPattern<nyacc::WhileOp> {
  using OpConversionPattern<nyacc::WhileOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      nyacc::WhileOp op, OpAdaptor adaptor [[maybe_unused]],
      mlir::ConversionPatternRewriter &rewriter) const override {
    // Clone the before and after regions
    auto& beforeRegion = op.getBefore();
    auto& afterRegion = op.getAfter();

    // Create the scf.while operation
    auto scfWhileOp = rewriter.create<mlir::scf::WhileOp>(
        op.getLoc(), mlir::TypeRange{}, mlir::ValueRange{});

    rewriter.inlineRegionBefore(beforeRegion, scfWhileOp.getBefore(),
                                scfWhileOp.getBefore().end());

    rewriter.inlineRegionBefore(afterRegion, scfWhileOp.getAfter(),
                                scfWhileOp.getAfter().end());

    // Erase the original operation
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

struct ConditionOpLowering : public mlir::OpConversionPattern<nyacc::ConditionOp> {
  using OpConversionPattern<nyacc::ConditionOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      nyacc::ConditionOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // Replace nyazy.condition with scf.condition
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        op, adaptor.getCondition(), mlir::ValueRange{});
    return mlir::success();
  }
};

struct YieldOpLowering : public mlir::OpConversionPattern<nyacc::YieldOp> {
  using OpConversionPattern<nyacc::YieldOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      nyacc::YieldOp op, OpAdaptor adaptor [[maybe_unused]],
      mlir::ConversionPatternRewriter &rewriter) const override {
    // Replace nyazy.yield with scf.yield
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op);
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

// Lowering for nyazy.print -> LLVM printf
struct PrintOpLowering : public mlir::OpConversionPattern<nyacc::PrintOp> {
  using OpConversionPattern<nyacc::PrintOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      nyacc::PrintOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    auto* ctx = module.getContext();

    // Get the LLVM ptr type
    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
    // auto elmType = mlir::IntegerType::get(ctx, 8);

    // Create format string global variable (if not existing)
    mlir::LLVM::GlobalOp formatStr;
    if (!(formatStr = module.lookupSymbol<mlir::LLVM::GlobalOp>("integer_format_specifier"))) {
      mlir::ConversionPatternRewriter::InsertionGuard guard(rewriter);
      mlir::Type charType = mlir::IntegerType::get(rewriter.getContext(), 8);
      mlir::Type arrayType = mlir::LLVM::LLVMArrayType::get(charType, 4); // "%ld\0"
      rewriter.setInsertionPointToStart(module.getBody());
      formatStr = rewriter.create<mlir::LLVM::GlobalOp>(
          loc, arrayType, /*isConstant=*/true, mlir::LLVM::Linkage::Internal,
          "integer_format_specifier", mlir::StringAttr::get(rewriter.getContext(), "%ld\n"));
    }

    // Get pointer to the format string
    mlir::Value formatStrPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, formatStr);
    mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(rewriter.getContext(), 64),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
    llvm::SmallVector<mlir::Value, 2> indices = {zero, zero};
    formatStrPtr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, ptrType, formatStr.getType(), formatStrPtr, indices);

    // Declare or get the printf function
    mlir::LLVM::LLVMFuncOp printfFunc;
    mlir::Type printfType = mlir::LLVM::LLVMFunctionType::get(
        mlir::IntegerType::get(rewriter.getContext(), 32),
        {ptrType}, /*isVarArg=*/true);
    if (!(printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))) {
      mlir::ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      printfFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
          loc, "printf", printfType);
    }

    // Convert the operand to LLVM type if necessary
    auto value = adaptor.getOperand();
    if (llvm::isa<mlir::IntegerType>(value.getType())) {
      // value = typeConverter->materializeTargetConversion(
      //     rewriter, loc, typeConverter->convertType(value.getType()), value);
    } else {
      // TODO: Support other types
      return mlir::failure();
    }

    // Call printf
    auto callop = rewriter.create<mlir::LLVM::CallOp>(
        loc,
        // printfType,
        mlir::TypeRange{mlir::IntegerType::get(rewriter.getContext(), 32)},
        mlir::SymbolRefAttr::get(printfFunc),
        mlir::ValueRange{formatStrPtr, value}
        );
    
    callop->setAttr("var_callee_type", mlir::TypeAttr::get(printfType));
    
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

} // namespace

void NyaZyToLLVMPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect>();
  target.addIllegalDialect<nyacc::NyaZyDialect>();
  target.addLegalOp<nyacc::PrintOp>();

  mlir::RewritePatternSet patterns(&getContext());
  // nyazy -> arith + func
  patterns.add<ConstantOpLowering, FuncOpLowering, ReturnOpLowering,
               AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering, 
               PosOpLowering, NegOpLowering, CmpOpLowering, CastOpLowering,
               AllocaOpLowering, LoadOpLowering, StoreOpLowering,
               ConditionOpLowering, YieldOpLowering, WhileOpLowering
               >(
      &getContext());

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

  // Lowering for nyazy.print -> LLVM printf
  mlir::ConversionTarget printTarget(getContext());
  printTarget.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
  printTarget.addIllegalOp<nyacc::PrintOp>();

  mlir::RewritePatternSet printPatterns(&getContext());
  printPatterns.add<PrintOpLowering>(&getContext());

  if (failed(
          applyFullConversion(getOperation(), printTarget, std::move(printPatterns)))) {
    signalPassFailure();
  }

}

std::unique_ptr<mlir::Pass> nyacc::createNyaZyToLLVMPass() {
  return std::make_unique<NyaZyToLLVMPass>();
}
