#include "parser.h"
#include "ast.h"
#include <charconv>
#include <iostream>
#include <memory>

namespace nyacc {

ModuleAST Parser::parseModule() {
  auto expr = parseExpr();
  return ModuleAST(std::move(expr));
}

// "1+1", "1"をparse
// "1+2+3"をparse
std::unique_ptr<ExprASTNode> Parser::parseExpr() {
  std::unique_ptr<ExprASTNode> node = parseMul();

  while (true) {
    const auto &token = tokens_[pos_];
    switch (token.getKind()) {
    case Token::TokenKind::Plus:
    case Token::TokenKind::Minus: {
      pos_++;
      auto rhs = parseMul();
      BinaryOp op = token.getKind() == Token::TokenKind::Plus ? BinaryOp::Add
                                                              : BinaryOp::Sub;
      node = std::make_unique<BinaryExpr>(std::move(node), std::move(rhs), op);
      continue;
    }
    default:
      return node;
    }
  }
}

std::unique_ptr<ExprASTNode> Parser::parseMul() {
  std::unique_ptr<ExprASTNode> node = parsePrimary();
  while (true) {
    const auto &token = tokens_[pos_];
    switch (token.getKind()) {
    case Token::TokenKind::Star:
    case Token::TokenKind::Slash: {
      pos_++;
      auto rhs = parseMul();
      BinaryOp op = token.getKind() == Token::TokenKind::Star ? BinaryOp::Mul
                                                              : BinaryOp::Div;
      node = std::make_unique<BinaryExpr>(std::move(node), std::move(rhs), op);
      continue;
    }
    default:
      return node;
    }
  }
  return node;
}

std::unique_ptr<ExprASTNode> Parser::parsePrimary() {
  const auto &token = tokens_[pos_];
  switch (token.getKind()) {
  case Token::TokenKind::NumLit: {
    int64_t result = 0;
    auto [ptr, ec] = std::from_chars(
        token.text().data(), token.text().data() + token.text().size(), result);
    if (ec == std::errc()) {
      pos_++;
      return std::make_unique<NumLitExpr>(result);
    } else {
      std::cerr << "Unexpected token: " << token << "\n";
      std::abort();
    }
  }
  default:
    std::cerr << "Unexpected token: " << token << "\n";
    std::abort();
    break;
  }
}

} // namespace nyacc