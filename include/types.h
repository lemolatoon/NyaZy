#pragma once
#include <cstddef>
#include <iostream>
namespace nyacc {

class Type {
public:
  enum class TypeKind {
    Primitive,
  };
  explicit Type(TypeKind kind) : kind_(kind) {}
  virtual ~Type() = default;

  TypeKind getKind() const { return kind_; }
  static const char *stringifyTypeKind(TypeKind);

private:
  TypeKind kind_;
};

class PrimitiveType : public Type {
public:
  enum class Kind {
    SInt,
  };
  PrimitiveType(Kind primitiveKind, size_t bitWidth)
      : Type(Type::TypeKind::Primitive), primitiveKind_(primitiveKind),
        bitWidth_(bitWidth) {}

  static bool classof(const Type *ty) {
    return ty->getKind() == TypeKind::Primitive;
  }

  Kind getPrimitiveKind() const { return primitiveKind_; }
  static const char *stringifyPrimitiveKind(Kind);
  size_t getBitWidth() const { return bitWidth_; }

private:
  Kind primitiveKind_;
  size_t bitWidth_;
};

} // namespace nyacc
std::ostream &operator<<(std::ostream &os, const nyacc::Type *type);
