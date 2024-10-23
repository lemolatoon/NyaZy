#include "parser.h"
#include "ast.h"
#include <iostream>
#include <charconv>

namespace nyacc {

ModuleAST Parser::parseModule() {
    auto expr = parseExpr();
    return ModuleAST(std::move(expr));
}

std::unique_ptr<ExprASTNode> Parser::parseExpr() {
    const auto &token = tokens_[pos_];
    switch (token.getKind()) {
    case Token::TokenKind::NumLit: {
        int64_t result = 0;
        auto [ptr, ec] = std::from_chars(token.text().data(), token.text().data() + token.text().size(), result);
    
        if (ec == std::errc()) {
          pos_++;
          return std::make_unique<NumLitExpr>(result);
        } else {
          std::cerr << "Unexpected token: " << token << "\n";
          std::abort();
        }
    }
    case Token::TokenKind::Eof:
      std::cerr << "Unexpected token: " << token << "\n";
      std::abort();
      break;
    }
}

} // nyacc