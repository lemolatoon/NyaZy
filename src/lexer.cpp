#include "lexer.h"
#include "tl/expected.hpp"
#include <cctype>
#include <error.h>
#include <iostream>
#include <sstream>
#include <string>

namespace nyacc {

std::ostream &operator<<(std::ostream &os, const Token &token) {
  os << "Token(" << Token::tokenKindToString(token.kind_) << ", " << token.text_
     << ")";
  return os;
}

std::string_view Lexer::head() { return input_.substr(pos_); }
void Lexer::advanceN(size_t n) {
  pos_ += n;
  currentLocation_.col += n;
}
void Lexer::advance() { advanceN(1); }
bool Lexer::atEof() const { return pos_ >= input_.size(); }
void Lexer::nextLine() {
  assert(input_[pos_] == '\n' &&
         "nextLine() must be called at the beginning of a line");
  pos_++;
  currentLocation_.line++;
  currentLocation_.col = 0;
}
tl::expected<std::vector<Token>, ErrorInfo> Lexer::tokenize() {
  std::vector<Token> tokens;

  while (!atEof()) {
    // tokenize integer
    if (std::isdigit(input_[pos_])) {
      const auto start_pos = pos_;
      auto loc = currentLocation();
      while (std::isdigit(input_[pos_])) {
        if (start_pos == pos_ && input_[pos_] == '0') {
          advance();
          break;
        }
        advance();
      }
      std::string_view num_lit = input_.substr(start_pos, pos_ - start_pos);
      tokens.emplace_back(Token::TokenKind::NumLit, num_lit, loc);
      continue;
    }

    if (input_[pos_] == ' ') {
      advance();
      continue;
    }

    if (input_[pos_] == '\n') {
      nextLine();
      continue;
    }

    const auto token_mapping = {
        std::pair<char, Token::TokenKind>{'+', Token::TokenKind::Plus},
        {'-', Token::TokenKind::Minus},
        {'*', Token::TokenKind::Star},
        {'/', Token::TokenKind::Slash},
        {'(', Token::TokenKind::OpenParen},
        {')', Token::TokenKind::CloseParen},
    };

    bool shouldContinue = false;
    for (const auto &[c, kind] : token_mapping) {
      if (input_[pos_] == c) {
        tokens.emplace_back(kind, input_.substr(pos_, 1), currentLocation());
        advance();
        shouldContinue = true;
        break;
      }
    }
    if (shouldContinue) {
      continue;
    }

    std::string error_msg;
    std::ostringstream oss;
    oss << "Unexpected character: " << input_[pos_];
    error_msg = oss.str();
    ErrorInfo info{.message = error_msg, .location = currentLocation()};
    return tl::unexpected{info};
  }
  tokens.emplace_back(Token::TokenKind::Eof, "", currentLocation());

  return tokens;
}

} // namespace nyacc