#pragma once

#include <string_view>
#include <vector>

namespace nyacc {
class Token {
public:
  enum class TokenKind {
    NumLit,
    Plus,
    Minus,
    Star,
    Slash,
    Eof,
  };
  static const char *tokenKindToString(TokenKind kind) {
    switch (kind) {
    case TokenKind::NumLit:
      return "NumLit";
    case TokenKind::Plus:
      return "Plus";
    case TokenKind::Minus:
      return "Minus";
    case TokenKind::Star:
      return "Star";
    case TokenKind::Slash:
      return "Slash";
    case TokenKind::Eof:
      return "Eof";
    }
  }
  Token(TokenKind kind, std::string_view text) : kind_(kind), text_(text) {}
  TokenKind getKind() const { return kind_; }
  std::string_view text() const { return text_; }

  friend std::ostream &operator<<(std::ostream &os, const Token &token);

private:
  TokenKind kind_;
  std::string_view text_;
};

class Lexer {
public:
  Lexer(std::string_view input) : input_(input), pos_(0) {}

  std::vector<Token> tokenize();
  std::string_view head();

private:
  std::string_view input_;
  size_t pos_;
};
} // namespace nyacc