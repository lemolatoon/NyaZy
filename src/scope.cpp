#include "scope.h"
#include "ast.h"
#include <iostream>

namespace nyacc {

std::optional<std::shared_ptr<DeclareStmt>>
Scope::lookup(const std::string &name) {
  if (auto v = localLookup(name)) {
    return *v;
  }

  if (parent_) {
    return parent_.value()->lookup(name);
  }

  return std::nullopt;
}

void Scope::insert(std::string name, std::shared_ptr<DeclareStmt> stmt) {
  ident_map_.insert_or_assign(name, stmt);
}

std::optional<std::shared_ptr<DeclareStmt>>
Scope::localLookup(const std::string &name) {
  auto it = ident_map_.find(name);
  if (it == ident_map_.end()) {
    return std::nullopt;
  }

  return it->second;
}

} // namespace nyacc
