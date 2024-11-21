#include "ast.h"
#include <iostream>

namespace nyacc {
void ModuleAST::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "ModuleAST\n";
  for (auto &stmt : getStmts()) {
    stmt->dump(level + 1);
  }
  std::cout << std::string(level * 2, ' ') << "Expr:\n";
  getExpr()->dump(level + 1);
}

void CallExpr::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "CallExpr(" << name_ << "\n";
  for (auto &arg : getArgs()) {
    arg->dump(level + 1);
  }
  std::cout << std::string(level * 2, ' ') << ")\n";
}

void BlockExpr::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "BlockExpr(\n";
  for (auto &stmt : getStmts()) {
    stmt->dump(level + 1);
  }
  std::cout << std::string(level * 2, ' ') << "Expr:\n";
  getExpr()->dump(level + 1);
  std::cout << std::string(level * 2, ' ') << ")\n";
}

void DeclareStmt::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "DeclareStmt(" << name_
            << " = \n";
  getInitExpr()->dump(level + 1);
  std::cout << std::string(level * 2, ' ') << ")\n";
}

void ExprStmt::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "ExprStmt(\n";
  getExpr()->dump(level + 1);
  std::cout << std::string(level * 2, ' ') << ")\n";
}

void WhileStmt::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "WhileStmt(\n";
  std::cout << std::string((level + 1) * 2, ' ') << "Cond:\n";
  getCond()->dump(level + 1);
  std::cout << std::string((level + 1) * 2, ' ') << "Body:\n";
  getBody()->dump(level + 1);
  std::cout << std::string(level * 2, ' ') << ")\n";
}

void NumLitExpr::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "NumLitExpr(" << value_ << ")\n";
}

void UnaryExpr::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "UnaryExpr(\n";
  std::cout << std::string((level + 1) * 2, ' ') << UnaryOpToStr(op_) << "\n";
  expr_->dump(level + 1);
  std::cout << std::string(level * 2, ' ') << ")\n";
}

void CastExpr::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "CastExpr(\n";
  expr_->dump(level + 1);
  std::cout << std::string((level + 1) * 2, ' ') << "->" << &getCastTo()
            << "\n";
  std::cout << std::string(level * 2, ' ') << ")\n";
}

void BinaryExpr::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "BinaryExpr(\n";
  lhs_->dump(level + 1);
  // print binary op
  std::cout << std::string((level + 1) * 2, ' ') << BinaryOpToStr(op_) << "\n";
  rhs_->dump(level + 1);
  std::cout << std::string(level * 2, ' ') << ")\n";
}

void AssignExpr::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "AssignExpr(\n";
  lhs_->dump(level + 1);
  std::cout << std::string((level + 1) * 2, ' ') << "=\n";
  rhs_->dump(level + 1);
  std::cout << std::string(level * 2, ' ') << ")\n";
}

void VariableExpr::dump(int level) const {
  std::cout << std::string(level * 2, ' ') << "VariableExpr(" << name_
            << ") -> \n";
  // getExpr()->dump(level + 1);
}
} // namespace nyacc
