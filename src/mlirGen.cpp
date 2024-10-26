#include "mlirGen.h"
#include "ast.h"
#include "ir/NyaZyDialect.h"
#include "ir/NyaZyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <iostream>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace {

class MLIRGenVisitor : public nyacc::Visitor {
public:
  MLIRGenVisitor(mlir::MLIRContext &context)
      : builder_(&context),
        module_(mlir::ModuleOp::create(builder_.getUnknownLoc())),
        value_(std::nullopt) {
    builder_.setInsertionPointToStart(module_.getBody());
  }

  mlir::OwningOpRef<mlir::ModuleOp> takeModule() { return std::move(module_); }

  void visit(const nyacc::ModuleAST &moduleAst) override {
    auto mainOp = builder_.create<nyacc::FuncOp>(
        builder_.getUnknownLoc(), "main", builder_.getFunctionType({}, {}));

    builder_.setInsertionPointToStart(&mainOp.front());

    builder_.setInsertionPointToStart(&mainOp.front());
    moduleAst.getExpr()->accept(*this);
    builder_.create<nyacc::ReturnOp>(builder_.getUnknownLoc(), value_.value());
  }

  void visit(const nyacc::NumLitExpr &numLit) override {
    value_ = builder_.create<nyacc::ConstantOp>(
        builder_.getUnknownLoc(),
        builder_.getI64IntegerAttr(numLit.getValue()));
  }

  // TODO
  void visit(const nyacc::BinaryExpr &binaryExpr) override {
    binaryExpr.getLhs()->accept(*this);
    auto lhs = value_.value();
    binaryExpr.getRhs()->accept(*this);
    auto rhs = value_.value();
    // 今はAddOpのみ

    switch (binaryExpr.getOp()) {
    case nyacc::BinaryOp::Add: {
      value_ =
          builder_.create<nyacc::AddOp>(builder_.getUnknownLoc(), lhs, rhs);
      break;
    }
    case nyacc::BinaryOp::Sub: {
      value_ =
          builder_.create<nyacc::SubOp>(builder_.getUnknownLoc(), lhs, rhs);
      break;
    }
    case nyacc::BinaryOp::Mul: {
      value_ =
          builder_.create<nyacc::MulOp>(builder_.getUnknownLoc(), lhs, rhs);
      break;
    }
    case nyacc::BinaryOp::Div: {
      value_ =
          builder_.create<nyacc::DivOp>(builder_.getUnknownLoc(), lhs, rhs);
      break;
    }
    }
  }

private:
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  std::optional<mlir::Value> value_;
};

} // namespace

namespace nyacc {

mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::gen(mlir::MLIRContext &context,
                                               const ModuleAST &moduleAst) {
  MLIRGenVisitor visitor{context};
  moduleAst.accept(visitor);
  return visitor.takeModule();
}

} // namespace nyacc