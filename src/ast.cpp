#include "ast.h"
#include <iostream>

namespace nyacc {
void ModuleAST::dump(int level) const {
  std::cout << "ModuleAST\n";
  expr_->dump(level + 1);
}

void NumLitExpr::dump(int level) const {
  std::cout << std::string(level, ' ') << "NumLitExpr(" << value_ << ")\n";
}

void BinaryExpr::dump(int level) const {
  std::cout << std::string(level, ' ') << "BinaryExpr(\n";
  lhs_->dump(level + 1);
  // print binary op
  std::cout << "+" << "\n"; // 後で実装
  rhs_->dump(level + 1);
  std::cout << std::string(level, ' ') << ")\n";
}
} // namespace nyacc