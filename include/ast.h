#pragma once

#include "expr.h"
#include "types.h"
#include <cstdint>
#include <error.h>
#include <vector>

namespace nyacc {
class Visitor {
public:
  virtual ~Visitor() = default;
  virtual void visit(const class ModuleAST &node) = 0;
  virtual void visit(const class DeclareStmt &node) = 0;
  virtual void visit(const class ExprStmt &node) = 0;
  virtual void visit(const class WhileStmt &node) = 0;
  virtual void visit(const class NumLitExpr &node) = 0;
  virtual void visit(const class BinaryExpr &node) = 0;
  virtual void visit(const class CastExpr &node) = 0;
  virtual void visit(const class UnaryExpr &node) = 0;
  virtual void visit(const class VariableExpr &node) = 0;
  virtual void visit(const class AssignExpr &node) = 0;
  virtual void visit(const class BlockExpr &node) = 0;
  virtual void visit(const class CallExpr &node) = 0;
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
    Block,
    Call,
  };
  explicit ExprASTNode(Location loc, ExprKind kind) : kind_(kind), loc_(loc) {}
  virtual ~ExprASTNode() = default;
  virtual void accept(class Visitor &v) = 0;
  virtual void dump(int level) const = 0;
  ExprKind getKind() const { return kind_; }
  Location getLoc() const { return loc_; }

private:
  ExprKind kind_;
  Location loc_;
};

class CallExpr : public ExprASTNode {
public:
  CallExpr(Location loc, std::string name, std::vector<Expr> args)
      : ExprASTNode(loc, ExprKind::Call), name_(std::move(name)),
        args_(std::move(args)) {}
  void accept(Visitor &v) override { v.visit(*this); }
  const std::string &getName() const { return name_; }
  const std::vector<Expr> &getArgs() const { return args_; }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Call;
  }

  void dump(int level) const override;

private:
  std::string name_;
  std::vector<Expr> args_;
};

class BlockExpr : public ExprASTNode {
public:
  BlockExpr(Location loc, std::vector<Stmt> stmts, Expr expr)
      : ExprASTNode(loc, ExprKind::Block), stmts_(std::move(stmts)),
        expr_(std::move(expr)) {}
  void accept(Visitor &v) override { v.visit(*this); }
  const std::vector<Stmt> &getStmts() const { return stmts_; }
  const Expr &getExpr() const { return expr_; }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Block;
  }

  void dump(int level) const override;

private:
  std::vector<Stmt> stmts_;
  Expr expr_;
};

class NumLitExpr : public ExprASTNode {
public:
  NumLitExpr(Location loc, int64_t value)
      : ExprASTNode(loc, ExprKind::NumLit), value_(value) {}

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
  UnaryExpr(Location loc, Expr expr, UnaryOp op)
      : ExprASTNode(loc, ExprKind::Unary), expr_(std::move(expr)), op_(op) {}

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
  BinaryExpr(Location loc, Expr lhs, Expr rhs, BinaryOp op)
      : ExprASTNode(loc, ExprKind::Binary), lhs_(std::move(lhs)),
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
  CastExpr(Location loc, Expr expr, PrimitiveType type)
      : ExprASTNode(loc, ExprKind::Cast), expr_(std::move(expr)), type_(type) {}
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
  AssignExpr(Location loc, Expr lhs, Expr rhs)
      : ExprASTNode(loc, ExprKind::Assign), lhs_(std::move(lhs)),
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
  VariableExpr(Location loc, std::string name,
               std::shared_ptr<DeclareStmt> declareStmt)
      : ExprASTNode(loc, ExprKind::Variable), name_(std::move(name)),
        declareStmt_(std::move(declareStmt)) {}
  void accept(Visitor &v) override { v.visit(*this); }

  static bool classof(const ExprASTNode *node) {
    return node->getKind() == ExprKind::Variable;
  }

  void dump(int level) const override;
  const std::shared_ptr<DeclareStmt> &getDeclareStmt() const {
    return declareStmt_;
  }
  const std::string &getName() const { return name_; }

private:
  std::string name_;
  std::shared_ptr<DeclareStmt> declareStmt_;
};

class StmtASTNode {
public:
  enum class StmtKind {
    Declare,
    Expr,
    While,
  };
  explicit StmtASTNode(Location loc, StmtKind kind) : kind_(kind), loc_(loc) {}
  virtual ~StmtASTNode() = default;
  virtual void accept(class Visitor &v) = 0;
  virtual void dump(int level) const = 0;
  StmtKind getKind() const { return kind_; }
  Location getLoc() const { return loc_; }

private:
  StmtKind kind_;
  Location loc_;
};

class DeclareStmt : public StmtASTNode {
public:
  DeclareStmt(Location loc, std::string name, Expr expr)
      : StmtASTNode(loc, StmtKind::Declare), name_(std::move(name)),
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
  ExprStmt(Location loc, Expr expr)
      : StmtASTNode(loc, StmtKind::Expr), expr_(std::move(expr)) {}
  static bool classof(const StmtASTNode *node) {
    return node->getKind() == StmtKind::Expr;
  }

  void dump(int level) const override;
  void accept(Visitor &v) override { v.visit(*this); }
  const Expr &getExpr() const { return expr_; }

private:
  Expr expr_;
};

class WhileStmt : public StmtASTNode {
public:
  WhileStmt(Location loc, Expr cond, Expr body)
      : StmtASTNode(loc, StmtKind::While), cond_(std::move(cond)),
        body_(std::move(body)) {}
  static bool classof(const StmtASTNode *node) {
    return node->getKind() == StmtKind::While;
  }

  void dump(int level) const override;
  void accept(Visitor &v) override { v.visit(*this); }
  const Expr &getCond() const { return cond_; }
  const Expr &getBody() const { return body_; }

private:
  Expr cond_;
  Expr body_;
};

class ModuleAST {
public:
  ModuleAST(Location loc, std::vector<Stmt> stmts, Expr expr)
      : stmts_(std::move(stmts)), expr_(expr), loc_(loc) {}
  void accept(Visitor &v) const { v.visit(*this); };
  void dump(int level = 0) const;
  const std::vector<Stmt> &getStmts() const { return stmts_; }
  const Expr &getExpr() const { return expr_; }
  const Location &getLoc() const { return loc_; }

private:
  std::vector<Stmt> stmts_;
  Expr expr_;
  Location loc_;
};

} // namespace nyacc
