#include "parser.h"
#include "ast.h"
#include <charconv>
#include <error.h>
#include <initializer_list>
#include <ios>
#include <iostream>
#include <llvm/Support/Casting.h>
#include <memory>

namespace nyacc {

Result<ModuleAST> Parser::parseModule() {
  auto loc = peek().getLoc();
  std::vector<Stmt> stmts;
  Expr lastExpr;
  while (!startsWith({Token::TokenKind::Eof})) {
    if (startsWithStmt()) {
      auto stmt = parseStmt();
      if (!stmt) {
        return tl::unexpected(stmt.error());
      }
      stmts.emplace_back(*stmt);
      continue;
    }

    auto expr = parseExpr();
    if (!expr) {
      return tl::unexpected{expr.error()};
    }
    if (!startsWith({Token::TokenKind::Semi})) {
      lastExpr = std::move(*expr);
      break;
    }
    pos_++;

    stmts.emplace_back(
        std::make_shared<ExprStmt>((*expr)->getLoc(), std::move(*expr)));
  }
  return ModuleAST(loc, std::move(stmts), std::move(lastExpr));
}

Result<Stmt> Parser::parseStmt() {
  if (startsWith({Token::TokenKind::Let})) {
    return parseDeclare();
  }
  if (startsWith({Token::TokenKind::While})) {
    return parseWhile();
  }
  auto expr = parseExpr();
  if (!expr) {
    return tl::unexpected{expr.error()};
  }
  EXPECT_EQ(peek().getLoc(), peek().getKind(), Token::TokenKind::Semi);
  pos_++;
  return std::make_shared<ExprStmt>((*expr)->getLoc(), std::move(*expr));
}

Result<Stmt> Parser::parseDeclare() {
  auto loc = peek().getLoc();
  EXPECT_EQ(peek().getLoc(), peek().getKind(), Token::TokenKind::Let);
  pos_++;

  EXPECT_EQ(peek().getLoc(), peek().getKind(), Token::TokenKind::Ident);
  auto name = std::string{tokens_[pos_].text()};
  pos_++;

  EXPECT_EQ(peek().getLoc(), peek().getKind(), Token::TokenKind::Eq);
  pos_++;

  auto expr = parseExpr();
  if (!expr) {
    return tl::unexpected{expr.error()};
  }

  EXPECT_EQ(peek().getLoc(), peek().getKind(), Token::TokenKind::Semi);
  pos_++;

  auto stmt = std::make_shared<DeclareStmt>(loc, name, std::move(*expr));
  scope_->insert(std::move(name), stmt);

  return stmt;
}

Result<Stmt> Parser::parseWhile() {
  auto loc = peek().getLoc();
  EXPECT_EQ(peek().getLoc(), peek().getKind(), Token::TokenKind::While);
  pos_++;

  auto cond = parseExpr();
  if (!cond) {
    return tl::unexpected{cond.error()};
  }

  auto body = parseExpr();
  if (!body) {
    return tl::unexpected{body.error()};
  }

  return std::make_shared<WhileStmt>(loc, std::move(*cond), std::move(*body));
}

Result<Expr> Parser::parseExpr() { return parseAssign(); }

Result<Expr> Parser::parseAssign() {
  auto loc = peek().getLoc();
  auto lhs = parseCompare();
  if (!lhs) {
    return lhs;
  }
  if (startsWith({Token::TokenKind::Eq})) {
    pos_++;
    EXPECT_TRUE(loc, llvm::isa<VariableExpr>(lhs->get()));
    auto rhs = parseCompare();
    if (!rhs) {
      return rhs;
    }
    auto var_expr = llvm::cast<VariableExpr>(lhs->get());
    EXPECT_TRUE(loc, scope_->lookup(var_expr->getName()).has_value());
    return std::make_shared<AssignExpr>(loc, std::move(*lhs), std::move(*rhs));
  }

  return lhs;
}

Result<Expr> Parser::parseCompare() {
  auto loc = peek().getLoc();
  auto lhs = parseAdd();
  if (!lhs) {
    return lhs;
  }

  while (true) {
    BinaryOp op;
    if (startsWith({Token::TokenKind::Eq, Token::TokenKind::Eq})) {
      pos_ += 2;
      op = BinaryOp::Eq;
    } else if (startsWith({Token::TokenKind::Gt, Token::TokenKind::Eq})) {
      pos_ += 2;
      op = BinaryOp::Gte;
    } else if (startsWith({Token::TokenKind::Gt})) {
      pos_ += 1;
      op = BinaryOp::Gt;
    } else if (startsWith({Token::TokenKind::Lt, Token::TokenKind::Eq})) {
      pos_ += 2;
      op = BinaryOp::Lte;
    } else if (startsWith({Token::TokenKind::Lt})) {
      pos_ += 1;
      op = BinaryOp::Lt;
    } else {
      return lhs;
    }
    auto rhs = parseAdd();
    if (!rhs) {
      return rhs;
    }
    lhs =
        std::make_shared<BinaryExpr>(loc, std::move(*lhs), std::move(*rhs), op);
    continue;
  }
}

Result<Expr> Parser::parseAdd() {
  auto loc = peek().getLoc();
  auto node = parseMul();
  if (!node) {
    return node;
  }

  while (true) {
    const auto &token = tokens_[pos_];
    switch (token.getKind()) {
    case Token::TokenKind::Plus:
    case Token::TokenKind::Minus: {
      pos_++;
      auto rhs = parseMul();
      if (!rhs) {
        return rhs;
      }
      BinaryOp op = token.getKind() == Token::TokenKind::Plus ? BinaryOp::Add
                                                              : BinaryOp::Sub;
      node = std::make_shared<BinaryExpr>(loc, std::move(*node),
                                          std::move(*rhs), op);
      continue;
    }
    default:
      return *node;
    }
  }
}

Result<Expr> Parser::parseMul() {
  auto loc = peek().getLoc();
  auto node = parseUnary();
  if (!node) {
    return node;
  }
  while (true) {
    const auto &token = tokens_[pos_];
    switch (token.getKind()) {
    case Token::TokenKind::Star:
    case Token::TokenKind::Slash: {
      pos_++;
      auto rhs = parseMul();
      if (!rhs) {
        return rhs;
      }
      BinaryOp op = token.getKind() == Token::TokenKind::Star ? BinaryOp::Mul
                                                              : BinaryOp::Div;
      node = std::make_shared<BinaryExpr>(loc, std::move(*node),
                                          std::move(*rhs), op);
      continue;
    }
    default:
      return *node;
    }
  }
}

Result<Expr> Parser::parseUnary() {
  const auto &token = peek();
  auto loc = token.getLoc();

  switch (token.getKind()) {
  case Token::TokenKind::Plus:
  case Token::TokenKind::Minus: {
    UnaryOp op = token.getKind() == Token::TokenKind::Plus ? UnaryOp::Plus
                                                           : UnaryOp::Minus;
    pos_++;
    auto expr = parsePostFix();
    if (!expr) {
      return expr;
    }
    return std::make_shared<UnaryExpr>(loc, std::move(*expr), op);
  }
  default:
    return parsePostFix();
  }
}

Result<Expr> Parser::parsePostFix() {
  auto loc = peek().getLoc();
  auto expr = parsePrimary();
  if (!expr) {
    return expr;
  }

  if (!startsWith({Token::TokenKind::As})) {
    return expr;
  }

  pos_++;

  const auto typeIdent = tokens_[pos_];
  EXPECT_EQ(typeIdent.getLoc(), typeIdent.getKind(), Token::TokenKind::Ident);
  pos_++;

  if (typeIdent.text()[0] != 'i') {
    return FATAL(typeIdent.getLoc(),
                 "Currently only types started with 'i' is supported but got ",
                 std::string{typeIdent.text()}, "\n");
  }

  size_t bitWidth;

  {
    auto sv = typeIdent.text().substr(1);
    auto [ptr, ec] =
        std::from_chars(sv.data(), sv.data() + sv.size(), bitWidth);

    if (ec != std::errc()) {
      return FATAL(typeIdent.getLoc(), "Parse BitWidth of Type failed",
                   std::string{sv}, "\n");
    }
  }

  return std::make_shared<CastExpr>(
      loc, std::move(*expr),
      PrimitiveType{PrimitiveType::Kind::SInt, bitWidth});
}

Result<Expr> Parser::parsePrimary() {
  const auto &token = peek();
  auto loc = token.getLoc();
  switch (token.getKind()) {
  case Token::TokenKind::NumLit: {
    int64_t result = 0;
    auto [ptr, ec] = std::from_chars(
        token.text().data(), token.text().data() + token.text().size(), result);
    if (ec == std::errc()) {
      pos_++;
      return std::make_shared<NumLitExpr>(loc, result);
    } else {
      return FATAL(loc, "Unexpected token: ", token, "\n");
    }
  }
  case Token::TokenKind::StrLit: {
    pos_++;
    return std::make_shared<StrLitExpr>(loc, std::string{token.text()});
  }
  case Token::TokenKind::OpenParen: {
    pos_++;
    auto expr = parseExpr();
    EXPECT_EQ(peek().getLoc(), peek().getKind(), Token::TokenKind::CloseParen);
    pos_++;
    return expr;
  }
  case Token::TokenKind::OpenBrace: {
    pos_++;
    std::vector<Stmt> stmts;
    Expr expr;
    while (!startsWith({Token::TokenKind::CloseBrace})) {
      if (startsWithStmt()) {
        auto stmt = parseStmt();
        if (!stmt) {
          return tl::unexpected{stmt.error()};
        }
        stmts.emplace_back(*stmt);
        continue;
      }

      auto expr_opt = parseExpr();
      if (!expr_opt) {
        return tl::unexpected{expr_opt.error()};
      }
      if (!startsWith({Token::TokenKind::Semi})) {
        expr = std::move(*expr_opt);
        break;
      }
      pos_++; // consume semi
      stmts.emplace_back(std::make_shared<ExprStmt>(loc, std::move(*expr_opt)));
    }

    if (!expr) {
      expr = std::make_shared<NumLitExpr>(loc, 0);
    }

    EXPECT_EQ(peek().getLoc(), peek().getKind(), Token::TokenKind::CloseBrace);
    pos_++;
    return std::make_shared<BlockExpr>(loc, std::move(stmts), std::move(expr));
  }
  case Token::TokenKind::Ident: {
    pos_++;
    std::string name{token.text()};
    if (startsWith({Token::TokenKind::OpenParen})) {
      pos_++;
      std::vector<Expr> args;
      while (!startsWith({Token::TokenKind::CloseParen})) {
        auto arg = parseExpr();
        if (!arg) {
          return tl::unexpected{arg.error()};
        }
        args.emplace_back(std::move(*arg));
        if (startsWith({Token::TokenKind::Comma})) {
          pos_++;
        }
      }
      pos_++;
      return std::make_shared<CallExpr>(loc, name, std::move(args));
    }
    auto declareStmt = scope_->lookup(name);
    if (!declareStmt) {
      return FATAL(token.getLoc(), "Variable '", name, "' not found. \n");
    }
    return std::make_shared<VariableExpr>(loc, name, *declareStmt);
  }
  default:
    return FATAL(token.getLoc(), "Unexpected token: ", token, "\n");
  }
}

const Token &Parser::peek() const { return tokens_[pos_]; }

bool Parser::startsWith(std::initializer_list<Token::TokenKind> tokens) const {
  if (pos_ + tokens.size() > tokens_.size()) {
    return false;
  }
  auto it = tokens.begin();
  for (size_t i = 0; i < tokens.size(); i++) {
    if (tokens_[pos_ + i].getKind() != *it) {
      return false;
    }
    it++;
  }
  return true;
}

bool Parser::startsWithStmt() const {
  return startsWith({Token::TokenKind::Let}) ||
         startsWith({Token::TokenKind::While});
}

} // namespace nyacc
