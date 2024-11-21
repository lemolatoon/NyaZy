#pragma once

#include "ast.h"
#include "expr.h"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace nyacc {
class Scope {
public:
  /// Constructor for the global scope
  explicit Scope() : parent_(std::nullopt), ident_map_() {}

  explicit Scope(std::shared_ptr<Scope> parent)
      : parent_(std::move(parent)), ident_map_() {}

  std::optional<std::shared_ptr<DeclareStmt>> lookup(const std::string &name);
  void insert(std::string name, std::shared_ptr<DeclareStmt> stmt);

private:
  std::optional<std::shared_ptr<DeclareStmt>>
  localLookup(const std::string &name);

  std::optional<std::shared_ptr<Scope>> parent_;
  std::unordered_map<std::string, std::shared_ptr<DeclareStmt>> ident_map_;
};
} // namespace nyacc
