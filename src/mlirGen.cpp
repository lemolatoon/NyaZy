#include "mlirGen.h"
#include "ast.h"
#include "ir/NyaZyDialect.h"
#include "ir/NyaZyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <error.h>
#include <iostream>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <variant>

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
        value_(std::nullopt), variableMap_(), flag_(std::monostate{}) {
    builder_.setInsertionPointToStart(module_.getBody());
  }

  nyacc::Result<mlir::OwningOpRef<mlir::ModuleOp>> takeModule() {
    if (auto e = error()) {
      return *e;
    }
    return std::move(module_);
  }

  mlir::Location mlirLoc(nyacc::Location loc) {
    return mlir::FileLineColLoc::get(builder_.getStringAttr(*loc.file),
                                     loc.line, loc.col);
  }

  void visit(const nyacc::ModuleAST &moduleAst) override {
    auto mainOp = builder_.create<nyacc::FuncOp>(
        mlirLoc(moduleAst.getLoc()), "main", builder_.getFunctionType({}, {}));

    builder_.setInsertionPointToStart(&mainOp.front());

    builder_.setInsertionPointToStart(&mainOp.front());
    for (auto &stmt : moduleAst.getStmts()) {
      stmt->accept(*this);
      if (has_error()) {
        return;
      }
    }
    moduleAst.getExpr()->accept(*this);
    if (has_error()) {
      return;
    }
    builder_.create<nyacc::ReturnOp>(mlirLoc(moduleAst.getExpr()->getLoc()),
                                     value_.value());
  }

  void visit(const class nyacc::DeclareStmt &node) override {
    node.getInitExpr()->accept(*this);
    if (has_error()) {
      return;
    }
    auto init = value_.value();
    auto memrefType = mlir::MemRefType::get({}, init.getType());
    auto memref =
        builder_.create<nyacc::AllocaOp>(mlirLoc(node.getLoc()), memrefType);
    builder_.create<nyacc::StoreOp>(mlirLoc(node.getLoc()), init, memref);
    variableMap_.insert({&node, memref});
  }
  void visit(const class nyacc::ExprStmt &node) override {
    node.getExpr()->accept(*this);
    if (has_error()) {
      return;
    }
  }

  void visit(const class nyacc::WhileStmt &node) override {
    auto loc = mlirLoc(node.getLoc());
    auto condLoc = mlirLoc(node.getCond()->getLoc());
    auto bodyLoc = mlirLoc(node.getBody()->getLoc());

    auto whileOp = builder_.create<nyacc::WhileOp>(loc);
    auto &beforeBlock = whileOp.getBefore().emplaceBlock();
    auto &afterBlock = whileOp.getAfter().emplaceBlock();

    // Condition
    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToEnd(&beforeBlock);
      node.getCond()->accept(*this);
      if (has_error()) {
        return;
      }
      auto cond = value_.value();
      builder_.create<nyacc::ConditionOp>(condLoc, cond);
    }

    // Body
    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToStart(&afterBlock);
      node.getBody()->accept(*this);
      if (has_error()) {
        return;
      }
      builder_.create<nyacc::YieldOp>(bodyLoc);
    }
  }

  void visit(const nyacc::NumLitExpr &numLit) override {
    value_ = builder_.create<nyacc::ConstantOp>(
        mlirLoc(numLit.getLoc()),
        builder_.getI64IntegerAttr(numLit.getValue()));
  }

  void visit(const nyacc::UnaryExpr &unaryExpr) override {
    auto loc = mlirLoc(unaryExpr.getLoc());
    unaryExpr.getExpr()->accept(*this);
    if (has_error()) {
      return;
    }
    auto expr = value_.value();
    switch (unaryExpr.getOp()) {
    case nyacc::UnaryOp::Plus: {
      value_ = builder_.create<nyacc::PosOp>(loc, expr);
      break;
    }
    case nyacc::UnaryOp::Minus: {
      value_ = builder_.create<nyacc::NegOp>(loc, expr);
      break;
    }
    }
  }

  void visit(const nyacc::CastExpr &castExpr) override {
    castExpr.getExpr()->accept(*this);
    if (has_error()) {
      return;
    }
    auto expr = value_.value();
    mlir::Type out =
        nyacc::asMLIRType(builder_.getContext(), castExpr.getCastTo());
    value_ =
        builder_.create<nyacc::CastOp>(mlirLoc(castExpr.getLoc()), out, expr);
  }

  // TODO
  void visit(const nyacc::BinaryExpr &binaryExpr) override {
    auto astLoc = binaryExpr.getLoc();
    auto loc = mlirLoc(astLoc);
    binaryExpr.getLhs()->accept(*this);
    if (has_error()) {
      return;
    }
    auto lhs = value_.value();
    binaryExpr.getRhs()->accept(*this);
    if (has_error()) {
      return;
    }
    auto rhs = value_.value();
    // 今はAddOpのみ

    switch (binaryExpr.getOp()) {
    case nyacc::BinaryOp::Add: {
      value_ = builder_.create<nyacc::AddOp>(loc, lhs, rhs);
      break;
    }
    case nyacc::BinaryOp::Sub: {
      value_ = builder_.create<nyacc::SubOp>(loc, lhs, rhs);
      break;
    }
    case nyacc::BinaryOp::Mul: {
      value_ = builder_.create<nyacc::MulOp>(loc, lhs, rhs);
      break;
    }
    case nyacc::BinaryOp::Div: {
      value_ = builder_.create<nyacc::DivOp>(loc, lhs, rhs);
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
        flag_ = FATAL(astLoc, "Unknown Comparison Operator",
                      static_cast<int>(binaryExpr.getOp()), "\n");
        return;
      }

      value_ = builder_.create<nyacc::CmpOp>(loc, pred, lhs, rhs);
    }
    }
  }
  void visit(const class nyacc::VariableExpr &node) override {
    auto memref = variableMap_.at(node.getDeclareStmt().get());
    value_ = builder_.create<nyacc::LoadOp>(mlirLoc(node.getLoc()), memref);
  }

  void visit(const class nyacc::AssignExpr &node) override {
    auto loc = node.getLoc();
    node.getRhs()->accept(*this);
    if (has_error()) {
      return;
    }
    auto rhs = value_.value();
    if (!llvm::isa<nyacc::VariableExpr>(node.getLhs().get())) {
      flag_ = FATAL(loc, "Left hand side of assignment must be a variable\n");
      return;
    }
    auto var_lhs = llvm::cast<nyacc::VariableExpr>(node.getLhs().get());
    auto memref = variableMap_.at(var_lhs->getDeclareStmt().get());

    builder_.create<nyacc::StoreOp>(mlirLoc(loc), rhs, memref);
  }

  void visit(const class nyacc::BlockExpr &node) override {
    for (auto &stmt : node.getStmts()) {
      stmt->accept(*this);
      if (has_error()) {
        return;
      }
    }
    node.getExpr()->accept(*this);
  }

private:
  bool has_error() { return !flag_.has_value(); }
  std::optional<tl::unexpected<nyacc::ErrorInfo>> error() {
    if (flag_.has_value()) {
      return std::nullopt;
    } else {
      return tl::unexpected{flag_.error()};
    }
  }
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  std::optional<mlir::Value> value_;
  std::map<const nyacc::DeclareStmt *, mlir::Value> variableMap_;
  nyacc::Result<std::monostate> flag_;
};

} // namespace

namespace nyacc {

Result<mlir::OwningOpRef<mlir::ModuleOp>>
MLIRGen::gen(mlir::MLIRContext &context, const ModuleAST &moduleAst) {
  MLIRGenVisitor visitor{context};
  moduleAst.accept(visitor);
  return visitor.takeModule();
}

} // namespace nyacc
