#include "ast.h"
#include <iostream>

namespace nyacc {
void ModuleAST::dump(int level) const {
  std::cout << "ModuleAST\n";
  for (auto &expr : getExprs()) {
    expr->dump(level + 1);
  }
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
  std::cout << std::string(level * 2, ' ') << "VariableExpr(" << name_ << ")\n";
}
} // namespace nyacc
