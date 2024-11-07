#pragma once

#include "expr.h"
#include "scope.h"
#include "types.h"
#include <cstdint>
#include <vector>

namespace nyacc {
class Visitor {
public:
  virtual ~Visitor() = default;
  virtual void visit(const class ModuleAST &node) = 0;
  virtual void visit(const class DeclareStmt &node) = 0;
  virtual void visit(const class ExprStmt &node) = 0;
  virtual void visit(const class NumLitExpr &node) = 0;
  virtual void visit(const class BinaryExpr &node) = 0;
  virtual void visit(const class CastExpr &node) = 0;
  virtual void visit(const class UnaryExpr &node) = 0;
  virtual void visit(const class VariableExpr &node) = 0;
  virtual void visit(const class AssignExpr &node) = 0;
};

class ExprASTNode {
public:
  enum class ExprKind {
    NumLit,
    Unary,
    Cast,
    Binary,
    Variable,
    Assign,
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
  UnaryExpr(Expr expr, UnaryOp op)
      : ExprASTNode(ExprKind::Unary), expr_(std::move(expr)), op_(op) {}

  void accept(Visitor &v) override { v.visit(*this); }

  void dump(int level) const override;
  const UnaryOp &getOp() const { return op_; }
  const Expr &getExpr() const { return expr_; }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Unary;
  }

private:
  Expr expr_;
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
  BinaryExpr(Expr lhs, Expr rhs, BinaryOp op)
      : ExprASTNode(ExprKind::Binary), lhs_(std::move(lhs)),
        rhs_(std::move(rhs)), op_(op) {}
  void accept(Visitor &v) override { v.visit(*this); }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Binary;
  }

  void dump(int level) const override;
  const Expr &getLhs() const { return lhs_; }
  const Expr &getRhs() const { return rhs_; }
  const BinaryOp &getOp() const { return op_; }

private:
  Expr lhs_;
  Expr rhs_;
  BinaryOp op_;
};

class CastExpr : public ExprASTNode {
public:
  CastExpr(Expr expr, PrimitiveType type)
      : ExprASTNode(ExprKind::Cast), expr_(std::move(expr)), type_(type) {}
  void accept(Visitor &v) override { v.visit(*this); }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Cast;
  }

  void dump(int level) const override;
  const Expr &getExpr() const { return expr_; }
  const PrimitiveType &getCastTo() const { return type_; }

private:
  Expr expr_;
  PrimitiveType type_;
};

class AssignExpr : public ExprASTNode {
public:
  AssignExpr(Expr lhs, Expr rhs)
      : ExprASTNode(ExprKind::Assign), lhs_(std::move(lhs)),
        rhs_(std::move(rhs)) {}
  void accept(Visitor &v) override { v.visit(*this); }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Assign;
  }

  void dump(int level) const override;
  const Expr &getLhs() const { return lhs_; }
  const Expr &getRhs() const { return rhs_; }

private:
  Expr lhs_;
  Expr rhs_;
};

class VariableExpr : public ExprASTNode {
public:
  VariableExpr(std::string name, Expr expr)
      : ExprASTNode(ExprKind::Variable), name_(std::move(name)),
        expr_(std::move(expr)) {}
  void accept(Visitor &v) override { v.visit(*this); }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Variable;
  }

  void dump(int level) const override;
  const Expr &getExpr() const { return expr_; }
  const std::string &getName() const { return name_; }

private:
  std::string name_;
  Expr expr_;
};

class StmtASTNode {
public:
  enum class StmtKind {
    Declare,
    Expr,
  };
  explicit StmtASTNode(StmtKind kind) : kind_(kind) {}
  virtual ~StmtASTNode() = default;
  virtual void accept(class Visitor &v) = 0;
  virtual void dump(int level) const = 0;
  StmtKind getKind() const { return kind_; };

private:
  StmtKind kind_;
};

class DeclareStmt : public StmtASTNode {
public:
  DeclareStmt(std::string name, Expr expr)
      : StmtASTNode(StmtKind::Declare), name_(std::move(name)),
        expr_(std::move(expr)) {}
  static bool classof(const StmtASTNode *node) {
    return node->getKind() == StmtKind::Declare;
  }

  void dump(int level) const override;
  void accept(Visitor &v) override { v.visit(*this); }
  const Expr &getInitExpr() const { return expr_; }
  const std::string &getName() const { return name_; }

private:
  std::string name_;
  Expr expr_;
};

class ExprStmt : public StmtASTNode {
public:
  ExprStmt(Expr expr) : StmtASTNode(StmtKind::Expr), expr_(std::move(expr)) {}
  static bool classof(const StmtASTNode *node) {
    return node->getKind() == StmtKind::Expr;
  }

  void dump(int level) const override;
  void accept(Visitor &v) override { v.visit(*this); }
  const Expr &getExpr() const { return expr_; }

private:
  Expr expr_;
};

class ModuleAST {
public:
  ModuleAST(std::vector<Stmt> stmts, Expr expr)
      : stmts_(std::move(stmts)), expr_(expr) {}
  void accept(Visitor &v) const { v.visit(*this); };
  void dump(int level = 0) const;
  const std::vector<Stmt> &getStmts() const { return stmts_; }
  const Expr &getExpr() const { return expr_; }

private:
  std::vector<Stmt> stmts_;
  Expr expr_;
};

} // namespace nyacc
