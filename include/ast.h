#pragma once

#include <cstdint>
#include <memory>

namespace nyacc {
class Visitor {
public:
  virtual ~Visitor() = default;
  virtual void visit(const class ModuleAST &node) = 0;
  virtual void visit(const class NumLitExpr &node) = 0;
  virtual void visit(const class BinaryExpr &node) = 0;
  virtual void visit(const class UnaryExpr &node) = 0;
};

class ExprASTNode {
public:
  enum class ExprKind {
    NumLit,
    Unary,
    Binary,
  };
  explicit ExprASTNode(ExprKind kind) : kind_(kind) {}
  virtual ~ExprASTNode() = default;
  virtual void accept(class Visitor &v) = 0;
  virtual void dump(int level) const = 0;
  ExprKind getKind() const { return kind_; };

private:
  ExprKind kind_;
};

class NumLitExpr : public ExprASTNode {
public:
  NumLitExpr(int64_t value) : ExprASTNode(ExprKind::NumLit), value_(value) {}

  void accept(Visitor &v) override { v.visit(*this); }
  int64_t getValue() const { return value_; }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::NumLit;
  }

  void dump(int level) const override;

private:
  int64_t value_;
};

enum class UnaryOp {
  Plus,
  Minus,
};

static inline const char *UnaryOpToStr(UnaryOp op) {
  switch (op) {
  case UnaryOp::Plus:
    return "+";
  case UnaryOp::Minus:
    return "-";
  }
}

class UnaryExpr : public ExprASTNode {
public:
  UnaryExpr(std::unique_ptr<ExprASTNode> expr, UnaryOp op)
      : ExprASTNode(ExprKind::Unary), expr_(std::move(expr)), op_(op) {}

  void accept(Visitor &v) override { v.visit(*this); }

  void dump(int level) const override;
  const UnaryOp &getOp() const { return op_; }
  const std::unique_ptr<ExprASTNode> &getExpr() const { return expr_; }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Unary;
  }

private:
  std::unique_ptr<ExprASTNode> expr_;
  UnaryOp op_;
};

enum class BinaryOp {
  Add,
  Sub,
  Mul,
  Div,
  Eq,
  Gte,
  Gt,
  Lte,
  Lt,
};

static inline const char *BinaryOpToStr(BinaryOp op) {
  switch (op) {
  case BinaryOp::Add:
    return "+";
  case BinaryOp::Sub:
    return "-";
  case BinaryOp::Mul:
    return "*";
  case BinaryOp::Div:
    return "/";
  case BinaryOp::Eq:
    return "==";
  case BinaryOp::Gte:
    return ">=";
  case BinaryOp::Gt:
    return ">";
  case BinaryOp::Lte:
    return "<=";
  case BinaryOp::Lt:
    return "<";
  }
}

class BinaryExpr : public ExprASTNode {
public:
  BinaryExpr(std::unique_ptr<ExprASTNode> lhs, std::unique_ptr<ExprASTNode> rhs,
             BinaryOp op)
      : ExprASTNode(ExprKind::Binary), lhs_(std::move(lhs)),
        rhs_(std::move(rhs)), op_(op) {}
  void accept(Visitor &v) override { v.visit(*this); }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Binary;
  }

  void dump(int level) const override;
  const std::unique_ptr<ExprASTNode> &getLhs() const { return lhs_; }
  const std::unique_ptr<ExprASTNode> &getRhs() const { return rhs_; }
  const BinaryOp &getOp() const { return op_; }

private:
  std::unique_ptr<ExprASTNode> lhs_;
  std::unique_ptr<ExprASTNode> rhs_;
  BinaryOp op_;
};

class ModuleAST {
public:
  ModuleAST(std::unique_ptr<ExprASTNode> expr) : expr_(std::move(expr)) {}
  void accept(Visitor &v) const { v.visit(*this); };
  void dump(int level = 0) const;
  const std::unique_ptr<ExprASTNode> &getExpr() const { return expr_; }

private:
  std::unique_ptr<ExprASTNode> expr_;
};
} // namespace nyacc