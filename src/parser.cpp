#include "parser.h"
#include "ast.h"
#include <charconv>
#include <initializer_list>
#include <iostream>
#include <llvm/Support/Casting.h>
#include <memory>

namespace nyacc {

ModuleAST Parser::parseModule() {
  std::vector<Stmt> stmts;
  Expr lastExpr;
  while (!startsWith({Token::TokenKind::Eof})) {
    if (startsWith({Token::TokenKind::Let})) {
      stmts.emplace_back(parseDeclare());
      continue;
    }

    auto expr = parseExpr();
    if (!startsWith({Token::TokenKind::Semi})) {
      lastExpr = std::move(expr);
      break;
    }
    pos_++;

    stmts.emplace_back(std::make_shared<ExprStmt>(std::move(expr)));
  }
  return ModuleAST(std::move(stmts), std::move(lastExpr));
}

Stmt Parser::parseDeclare() {
  assert(startsWith({Token::TokenKind::Let}));
  pos_++;

  assert(startsWith({Token::TokenKind::Ident}));
  auto name = std::string{tokens_[pos_].text()};
  pos_++;

  assert(startsWith({Token::TokenKind::Eq}));
  pos_++;

  auto expr = parseExpr();

  assert(startsWith({Token::TokenKind::Semi}));
  pos_++;

  scope_->insert(name, expr);
  return std::make_shared<DeclareStmt>(std::move(name), std::move(expr));
}

Expr Parser::parseExpr() { return parseAssign(); }

Expr Parser::parseAssign() {
  auto lhs = parseCompare();
  if (startsWith({Token::TokenKind::Eq})) {
    pos_++;
    assert(llvm::isa<VariableExpr>(lhs.get()));
    auto rhs = parseCompare();
    auto var_expr = llvm::cast<VariableExpr>(lhs.get());
    scope_->insert(var_expr->getName(), rhs);
    return std::make_shared<AssignExpr>(std::move(lhs), std::move(rhs));
  }

  return lhs;
}

Expr Parser::parseCompare() {
  auto lhs = parseAdd();

  while (true) {
    BinaryOp op;
    if (startsWith({Token::TokenKind::Eq, Token::TokenKind::Eq})) {
      pos_ += 2;
      op = BinaryOp::Eq;
    } else if (startsWith({Token::TokenKind::Gt, Token::TokenKind::Eq})) {
      pos_ += 2;
      op = BinaryOp::Gte;
    } else if (startsWith({Token::TokenKind::Gt})) {
      pos_ += 1;
      op = BinaryOp::Gt;
    } else if (startsWith({Token::TokenKind::Lt, Token::TokenKind::Eq})) {
      pos_ += 2;
      op = BinaryOp::Lte;
    } else if (startsWith({Token::TokenKind::Lt})) {
      pos_ += 1;
      op = BinaryOp::Lt;
    } else {
      return lhs;
    }
    auto rhs = parseAdd();
    lhs = std::make_shared<BinaryExpr>(std::move(lhs), std::move(rhs), op);
    continue;
  }
}

Expr Parser::parseAdd() {
  Expr node = parseMul();

  while (true) {
    const auto &token = tokens_[pos_];
    switch (token.getKind()) {
    case Token::TokenKind::Plus:
    case Token::TokenKind::Minus: {
      pos_++;
      auto rhs = parseMul();
      BinaryOp op = token.getKind() == Token::TokenKind::Plus ? BinaryOp::Add
                                                              : BinaryOp::Sub;
      node = std::make_shared<BinaryExpr>(std::move(node), std::move(rhs), op);
      continue;
    }
    default:
      return node;
    }
  }
}

Expr Parser::parseMul() {
  Expr node = parseUnary();
  while (true) {
    const auto &token = tokens_[pos_];
    switch (token.getKind()) {
    case Token::TokenKind::Star:
    case Token::TokenKind::Slash: {
      pos_++;
      auto rhs = parseMul();
      BinaryOp op = token.getKind() == Token::TokenKind::Star ? BinaryOp::Mul
                                                              : BinaryOp::Div;
      node = std::make_shared<BinaryExpr>(std::move(node), std::move(rhs), op);
      continue;
    }
    default:
      return node;
    }
  }
  return node;
}

Expr Parser::parseUnary() {
  const auto &token = tokens_[pos_];

  switch (token.getKind()) {
  case Token::TokenKind::Plus:
  case Token::TokenKind::Minus: {
    UnaryOp op = token.getKind() == Token::TokenKind::Plus ? UnaryOp::Plus
                                                           : UnaryOp::Minus;
    pos_++;
    auto expr = parsePostFix();
    return std::make_shared<UnaryExpr>(std::move(expr), op);
  }
  default:
    return parsePostFix();
  }
}

Expr Parser::parsePostFix() {
  auto expr = parsePrimary();

  if (!startsWith({Token::TokenKind::As})) {
    return expr;
  }

  pos_++;

  const auto typeIdent = tokens_[pos_];
  if (typeIdent.getKind() != Token::TokenKind::Ident) {
    std::cerr << "Expected Ident but got "
              << Token::tokenKindToString(typeIdent.getKind()) << std::endl;
    std::abort();
  }
  pos_++;

  if (typeIdent.text()[0] != 'i') {
    std::cerr << "Currently only types started with 'i' is supported but got "
              << std::string{typeIdent.text()} << std::endl;
  }

  size_t bitWidth;

  {
    auto sv = typeIdent.text().substr(1);
    auto [ptr, ec] =
        std::from_chars(sv.data(), sv.data() + sv.size(), bitWidth);

    if (ec != std::errc()) {
      std::cerr << "Parse BitWidth of Type failed" << std::string{sv}
                << std::endl;
    }
  }

  return std::make_shared<CastExpr>(
      std::move(expr), PrimitiveType{PrimitiveType::Kind::SInt, bitWidth});
}

Expr Parser::parsePrimary() {
  const auto &token = tokens_[pos_];
  switch (token.getKind()) {
  case Token::TokenKind::NumLit: {
    int64_t result = 0;
    auto [ptr, ec] = std::from_chars(
        token.text().data(), token.text().data() + token.text().size(), result);
    if (ec == std::errc()) {
      pos_++;
      return std::make_shared<NumLitExpr>(result);
    } else {
      std::cerr << "Unexpected token: " << token << "\n";
      std::abort();
    }
  }
  case Token::TokenKind::OpenParen: {
    pos_++;
    auto expr = parseExpr();
    if (tokens_[pos_].getKind() != Token::TokenKind::CloseParen) {
      std::cerr << "Expected ')'\n";
      std::abort();
    }
    pos_++;
    return expr;
  }
  case Token::TokenKind::Ident: {
    pos_++;
    std::string name{token.text()};
    auto expr = scope_->lookup(name);
    if (!expr) {
      std::cerr << "Variable '" << name << "' not found. ";
      std::abort();
    }
    return std::make_shared<VariableExpr>(name, *expr);
  }
  default:
    std::cerr << "Unexpected token: " << token << "\n";
    std::abort();
    break;
  }
}

bool Parser::startsWith(std::initializer_list<Token::TokenKind> tokens) const {
  if (pos_ + tokens.size() > tokens_.size()) {
    return false;
  }
  auto it = tokens.begin();
  for (size_t i = 0; i < tokens.size(); i++) {
    if (tokens_[pos_ + i].getKind() != *it) {
      return false;
    }
    it++;
  }
  return true;
}

} // namespace nyacc
