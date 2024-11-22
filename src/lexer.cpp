#include "lexer.h"
#include "tl/expected.hpp"
#include <algorithm>
#include <cctype>
#include <error.h>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>

namespace nyacc {

std::ostream &operator<<(std::ostream &os, const Token &token) {
  os << "Token(" << Token::tokenKindToString(token.kind_) << ", " << token.text_
     << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Token::TokenKind &kind) {
  os << Token::tokenKindToString(kind);
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
Result<std::vector<Token>> Lexer::tokenize() {
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

    if (startsWith("\"")) {
      auto loc = currentLocation();
      advance();
      const auto start_pos = pos_;
      while (!atEof() && input_[pos_] != '"') {
        advance();
      }
      if (atEof()) {
        return FATAL(loc, "Unterminated string literal");
      }
      std::string_view str_lit = input_.substr(start_pos, pos_ - start_pos);
      advance();
      tokens.emplace_back(Token::TokenKind::StrLit, str_lit, loc);
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
        {'{', Token::TokenKind::OpenBrace},
        {'}', Token::TokenKind::CloseBrace},
        {';', Token::TokenKind::Semi},
        {',', Token::TokenKind::Comma}};

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

    const auto isPanct = [&](char c) -> bool {
      return std::ranges::any_of(token_mapping,
                                 [&](auto &pair) { return c == pair.first; });
    };

    const auto long_token_mapping = {
        std::pair<std::string_view, Token::TokenKind>{"as",
                                                      Token::TokenKind::As},
        {"let", Token::TokenKind::Let},
        {"while", Token::TokenKind::While}};

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

    if (!atEof() && !startsWithSpace(true) && !isPanct(input_[pos_])) {
      const size_t startPos = pos_;
      auto loc = currentLocation();
      advance();
      while (!atEof() && !startsWithSpace(true) && !isPanct(input_[pos_])) {
        advance();
      }
      tokens.emplace_back(Token::TokenKind::Ident,
                          input_.substr(startPos, pos_ - startPos), loc);
      continue;
    }

    return FATAL(currentLocation(), "Unexpected character: ", input_[pos_]);
  }
  tokens.emplace_back(Token::TokenKind::Eof, "", currentLocation());

  return tokens;
}

} // namespace nyacc
