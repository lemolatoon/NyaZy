#ifndef NYAZY_BASE
#define NYAZY_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def NyaZy_CmpPredicateAttr : I64EnumAttr<
    "CmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"lt", 2>,
      I64EnumAttrCase<"le", 3>,
      I64EnumAttrCase<"gt", 4>,
      I64EnumAttrCase<"ge", 5>,
    ]> {
  let cppNamespace = "::nyacc";
}

#endif // NYAZY_BASE