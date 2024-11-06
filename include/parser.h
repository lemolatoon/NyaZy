#pragma once

#include "ast.h"
#include "lexer.h"
#include "scope.h"

namespace nyacc {
class Parser {
public:
  Parser(std::vector<Token> tokens)
      : tokens_(std::move(tokens)), pos_(0),
        global_scope_(std::make_shared<Scope>()), scope_(global_scope_) {}

  ModuleAST parseModule();

private:
  Expr parseExpr();
  Expr parseCompare();
  Expr parseAdd();
  Expr parseMul();
  Expr parseUnary();
  Expr parsePostFix();
  Expr parsePrimary();

  bool startsWith(std::initializer_list<Token::TokenKind> list) const;
  std::vector<Token> tokens_;
  size_t pos_{0};

  std::shared_ptr<Scope> global_scope_;
  std::shared_ptr<Scope> scope_;
};
} // namespace nyacc
