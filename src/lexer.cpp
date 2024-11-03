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
bool Lexer::startsWithSpace(bool includesNewline) const {
  char c = input_[pos_];
  return (includesNewline && c == '\n') || c == '\t' || c == ' ' || c == '\r';
}
bool Lexer::startsWith(std::string_view sv) const {
  if (pos_ + sv.size() > input_.size()) {
    return false;
  }

  for (size_t i = 0; i < sv.size(); i++) {
    if (input_[pos_ + i] != sv[i]) {
      return false;
    }
  }

  return true;
}
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
      while (!atEof() && std::isdigit(input_[pos_])) {
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

    if (input_[pos_] == '\n') {
      nextLine();
      continue;
    }

    if (startsWithSpace(false)) {
      advance();
      continue;
    }

    const auto token_mapping = {
        std::pair<char, Token::TokenKind>{'+', Token::TokenKind::Plus},
        {'-', Token::TokenKind::Minus},
        {'*', Token::TokenKind::Star},
        {'/', Token::TokenKind::Slash},
        {'(', Token::TokenKind::OpenParen},
        {')', Token::TokenKind::CloseParen},
        {'=', Token::TokenKind::Eq},
        {'>', Token::TokenKind::Gt},
        {'<', Token::TokenKind::Lt},
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

    const auto long_token_mapping = {
        std::pair<std::string_view, Token::TokenKind>{"as",
                                                      Token::TokenKind::As},
    };

    for (const auto &[c, kind] : long_token_mapping) {
      if (startsWith(c)) {
        tokens.emplace_back(kind, input_.substr(pos_, c.size()),
                            currentLocation());
        advanceN(c.size());
        shouldContinue = true;
        break;
      }
    }
    if (shouldContinue) {
      continue;
    }

    if (!atEof() && !startsWithSpace(true)) {
      const size_t startPos = pos_;
      advance();
      while (!atEof() && !startsWithSpace(true)) {
        advance();
      }
      tokens.emplace_back(Token::TokenKind::Ident,
                          input_.substr(startPos, pos_ - startPos),
                          currentLocation());
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
