#include "ast.h"
#include <iostream>

namespace nyacc {
void ModuleAST::dump(int level) const {
  std::cout << "ModuleAST\n";
  expr_->dump(level + 1);
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
} // namespace nyacc
