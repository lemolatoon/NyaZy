#pragma once

#include "error.h"
#include <cassert>
#include <expected>
#include <memory>
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
    OpenParen,
    CloseParen,
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
    case TokenKind::OpenParen:
      return "OpenParen";
    case TokenKind::CloseParen:
      return "CloseParen";
    case TokenKind::Eof:
      return "Eof";
    }
  }
  Token(TokenKind kind, std::string_view text, Location loc)
      : kind_(kind), text_(text), loc_(loc) {}
  TokenKind getKind() const { return kind_; }
  std::string_view text() const { return text_; }

  friend std::ostream &operator<<(std::ostream &os, const Token &token);

private:
  TokenKind kind_;
  std::string_view text_;
  Location loc_;
};

class Lexer {
public:
  Lexer(std::string_view input)
      : Lexer(input, std::make_shared<std::string>("unkown-file")) {}
  Lexer(std::string_view input, std::shared_ptr<std::string> filename)
      : input_(input), pos_(0),
        currentLocation_(Location{.file = filename, .line = 0, .col = 0}) {}

  std::expected<std::vector<Token>, ErrorInfo> tokenize();
  const Location &currentLocation() const { return currentLocation_; }

private:
  std::string_view head();

  void advanceN(size_t n);

  void advance();

  bool atEof() const;

  void nextLine();

  std::string_view input_;
  size_t pos_;

  Location currentLocation_{};
};
} // namespace nyacc