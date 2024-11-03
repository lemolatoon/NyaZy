#include "types.h"
#include <iostream>
#include <llvm/Support/Casting.h>
namespace nyacc {
const char *Type::stringifyTypeKind(TypeKind kind) {
  switch (kind) {
  case TypeKind::Primitive:
    return "Primitive";
  }
}

const char *PrimitiveType::stringifyPrimitiveKind(Kind kind) {
  switch (kind) {
  case Kind::SInt:
    return "SInt";
  }
}

} // namespace nyacc

std::ostream &operator<<(std::ostream &os, const nyacc::Type *type) {
  os << nyacc::Type::stringifyTypeKind(type->getKind()) << ":";
  switch (type->getKind()) {
  case nyacc::Type::TypeKind::Primitive: {
    auto *pt = llvm::cast<nyacc::PrimitiveType>(type);
    os << nyacc::PrimitiveType::stringifyPrimitiveKind(pt->getPrimitiveKind());
    os << ", " << pt->getBitWidth() << "-bit";
  }
  }
  return os;
}
