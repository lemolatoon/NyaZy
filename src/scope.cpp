#include "scope.h"
#include <iostream>

namespace nyacc {

std::optional<Expr> Scope::lookup(const std::string &name) {
  if (auto v = localLookup(name)) {
    return *v;
  }

  if (parent_) {
    return parent_.value()->lookup(name);
  }

  return std::nullopt;
}

void Scope::insert(std::string name, Expr expr) {
  ident_map_.insert_or_assign(name, expr);
}

std::optional<Expr> Scope::localLookup(const std::string &name) {
  auto it = ident_map_.find(name);
  if (it == ident_map_.end()) {
    return std::nullopt;
  }

  return it->second;
}

} // namespace nyacc
