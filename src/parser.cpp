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
std::unique_ptr<ExprASTNode> Parser::parseExpr() {
  std::unique_ptr<ExprASTNode> node = parsePrimary();

  const auto &token = tokens_[pos_];
  switch (token.getKind()) {
  case Token::TokenKind::Plus: {
    pos_++;
    auto rhs = parsePrimary();
    return std::make_unique<BinaryExpr>(std::move(node), std::move(rhs),
                                        BinaryOp::Add);
  }
  default:
    return node;
  }
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