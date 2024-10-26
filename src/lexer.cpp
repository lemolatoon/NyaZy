#include "lexer.h"
#include <cctype>
#include <iostream>

namespace nyacc {

std::ostream &operator<<(std::ostream &os, const Token &token) {
  os << "Token(" << Token::tokenKindToString(token.kind_) << ", " << token.text_
     << ")";
  return os;
}

std::string_view Lexer::head() { return input_.substr(pos_); }
std::vector<Token> Lexer::tokenize() {
  std::vector<Token> tokens;

  while (pos_ < input_.size()) {
    // tokenize integer
    if (std::isdigit(input_[pos_])) {
      const auto start_pos = pos_;
      while (std::isdigit(input_[pos_])) {
        if (start_pos == pos_ && input_[pos_] == '0') {
          pos_++;
          break;
        }
        pos_++;
      }
      std::string_view num_lit = input_.substr(start_pos, pos_ - start_pos);
      tokens.emplace_back(Token::TokenKind::NumLit, num_lit);
      continue;
    }

    if (input_[pos_] == ' ' || input_[pos_] == '\n') {
      pos_++;
      continue;
    }

    // tokenize plus
    if (input_[pos_] == '+') {
      tokens.emplace_back(Token::TokenKind::Plus, "+");
      pos_++;
      continue;
    }

    // tokenize minus
    if (input_[pos_] == '-') {
      tokens.emplace_back(Token::TokenKind::Minus, "-");
      pos_++;
      continue;
    }

    // tokenize mul
    if (input_[pos_] == '*') {
      tokens.emplace_back(Token::TokenKind::Star, "*");
      pos_++;
      continue;
    }

    // tokenize slash
    if (input_[pos_] == '/') {
      tokens.emplace_back(Token::TokenKind::Slash, "/");
      pos_++;
      continue;
    }
  }

  return tokens;
}

} // namespace nyacc