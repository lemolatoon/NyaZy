#include "mlirGen.h"
#include "ast.h"
#include "ir/NyaZyDialect.h"
#include "ir/NyaZyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <iostream>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>

namespace nyacc {
mlir::Type asMLIRType(mlir::MLIRContext *ctx, Type type) {
  switch (type.getKind()) {
  case Type::TypeKind::Primitive:
    return asMLIRType(ctx, llvm::cast<PrimitiveType>(type));
  }
}

mlir::Type asMLIRType(mlir::MLIRContext *ctx, PrimitiveType type) {
  switch (type.getPrimitiveKind()) {
  case PrimitiveType::Kind::SInt:
    return mlir::IntegerType::get(ctx, type.getBitWidth());
  }
}

} // namespace nyacc

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

  void visit(const nyacc::UnaryExpr &unaryExpr) override {
    unaryExpr.getExpr()->accept(*this);
    auto expr = value_.value();
    switch (unaryExpr.getOp()) {
    case nyacc::UnaryOp::Plus: {
      value_ = builder_.create<nyacc::PosOp>(builder_.getUnknownLoc(), expr);
      break;
    }
    case nyacc::UnaryOp::Minus: {
      value_ = builder_.create<nyacc::NegOp>(builder_.getUnknownLoc(), expr);
      break;
    }
    }
  }

  void visit(const nyacc::CastExpr &castExpr) override {
    castExpr.getExpr()->accept(*this);
    auto expr = value_.value();
    mlir::Type out =
        nyacc::asMLIRType(builder_.getContext(), castExpr.getCastTo());
    value_ =
        builder_.create<nyacc::CastOp>(builder_.getUnknownLoc(), out, expr);
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
    case nyacc::BinaryOp::Eq:
    case nyacc::BinaryOp::Gte:
    case nyacc::BinaryOp::Gt:
    case nyacc::BinaryOp::Lte:
    case nyacc::BinaryOp::Lt: {
      nyacc::CmpPredicate pred;
      switch (binaryExpr.getOp()) {
      case nyacc::BinaryOp::Eq:
        pred = nyacc::CmpPredicate::eq;
        break;
      case nyacc::BinaryOp::Gte:
        pred = nyacc::CmpPredicate::ge;
        break;
      case nyacc::BinaryOp::Gt:
        pred = nyacc::CmpPredicate::gt;
        break;
      case nyacc::BinaryOp::Lte:
        pred = nyacc::CmpPredicate::le;
        break;
      case nyacc::BinaryOp::Lt:
        pred = nyacc::CmpPredicate::lt;
        break;
      default:
        std::cerr << "Unknown Comparison Operator\n";
        std::abort();
      }

      value_ = builder_.create<nyacc::CmpOp>(builder_.getUnknownLoc(), pred,
                                             lhs, rhs);
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
