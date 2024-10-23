#pragma once

#include <string_view>
#include <vector>

namespace nyacc {
    class Token {
        public:
            enum class TokenKind {
                NumLit,
                Eof,
            };
            static const char *tokenKindToString(TokenKind kind) {
                switch (kind) {
                    case TokenKind::NumLit:
                        return "NumLit";
                    case TokenKind::Eof:
                        return "Eof";
                }
            }
            Token(TokenKind kind, std::string_view text) : kind_(kind), text_(text) {}
            TokenKind getKind() const {
                return kind_;
            }

            friend std::ostream& operator<<(std::ostream &os, const Token &token);
        private:
            TokenKind kind_;
            std::string_view text_;
    };

    class Lexer {
        public:
            Lexer(std::string_view input): input_(input), pos_(0) {}

            std::vector<Token> tokenize();
            std::string_view head();
        private:
            std::string_view input_;
            size_t pos_;
    };
}