#pragma once

#include "ast.h"
#include "lexer.h"

namespace nyacc {
class Parser {
public:
  Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)), pos_(0) {}

  ModuleAST parseModule();

private:
  std::unique_ptr<ExprASTNode> parseExpr();
  std::unique_ptr<ExprASTNode> parsePrimary();
  std::unique_ptr<ExprASTNode> parseMul();
  std::vector<Token> tokens_;
  size_t pos_{0};
};
} // namespace nyacc