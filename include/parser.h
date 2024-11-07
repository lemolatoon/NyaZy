#pragma once

#include "ast.h"
#include "error.h"
#include "lexer.h"
#include "scope.h"

namespace nyacc {
class Parser {
public:
  Parser(std::vector<Token> tokens)
      : tokens_(std::move(tokens)), pos_(0),
        global_scope_(std::make_shared<Scope>()), scope_(global_scope_) {}

  Result<ModuleAST> parseModule();

private:
  Result<Stmt> parseDeclare();

  Result<Expr> parseExpr();
  Result<Expr> parseAssign();
  Result<Expr> parseCompare();
  Result<Expr> parseAdd();
  Result<Expr> parseMul();
  Result<Expr> parseUnary();
  Result<Expr> parsePostFix();
  Result<Expr> parsePrimary();

  bool startsWith(std::initializer_list<Token::TokenKind> list) const;
  const Token &peek() const;
  std::vector<Token> tokens_;
  size_t pos_{0};

  std::shared_ptr<Scope> global_scope_;
  std::shared_ptr<Scope> scope_;
};
} // namespace nyacc
