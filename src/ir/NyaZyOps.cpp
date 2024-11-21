#include "ir/NyaZyOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/DialectImplementation.h"

namespace nyacc {

std::optional<CmpPredicate> CmpOp::getPredicateByName(mlir::StringRef name) {
  if (name == "eq")
    return CmpPredicate::eq;
  if (name == "ne")
    return CmpPredicate::ne;
  if (name == "lt")
    return CmpPredicate::lt;
  if (name == "le")
    return CmpPredicate::le;
  if (name == "gt")
    return CmpPredicate::gt;
  if (name == "ge")
    return CmpPredicate::ge;

  return std::nullopt;
}

/// CastOp only accepts casts between primitive types
bool CastOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }
  auto input = inputs.front();
  auto output = outputs.front();
  return input.isIntOrFloat() && output.isIntOrFloat();
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

/// Parses a `while` op.
///
/// op ::= `scf.while` assignments `:` function-type region `do` region
///         `attributes` attribute-dict
/// initializer ::= /* empty */ | `(` assignment-list `)`
/// assignment-list ::= assignment | assignment `,` assignment-list
/// assignment ::= ssa-value `=` ssa-value
mlir::ParseResult nyacc::WhileOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::SmallVector<mlir::OpAsmParser::Argument, 4> regionArgs;
  mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  mlir::Region *before = result.addRegion();
  mlir::Region *after = result.addRegion();

  // mlir::OptionalParseResult listResult =
  //     parser.parseOptionalAssignmentList(regionArgs, operands);
  // if (listResult.has_value() && failed(listResult.value()))
  //   return mlir::failure();

  // mlir::FunctionType functionType;
  // mlir::SMLoc typeLoc = parser.getCurrentLocation();
  // if (failed(parser.parseColonType(functionType)))
  //   return mlir::failure();

  // result.addTypes(functionType.getResults());

  // if (functionType.getNumInputs() != operands.size()) {
  //   return parser.emitError(typeLoc)
  //          << "expected as many input types as operands "
  //          << "(expected " << operands.size() << " got "
  //          << functionType.getNumInputs() << ")";
  // }

  // // Resolve input operands.
  // if (failed(parser.resolveOperands(operands, functionType.getInputs(),
  //                                   parser.getCurrentLocation(),
  //                                   result.operands)))
  //   return mlir::failure();

  // // Propagate the types into the region arguments.
  // for (size_t i = 0, e = regionArgs.size(); i != e; ++i)
  //   regionArgs[i].type = functionType.getInput(i);

  return mlir::failure(parser.parseRegion(*before, regionArgs) ||
                 parser.parseKeyword("do") || parser.parseRegion(*after) ||
                 parser.parseOptionalAttrDictWithKeyword(result.attributes));
}

/// Prints a `while` op.
void nyacc::WhileOp::print(mlir::OpAsmPrinter &p) {
  // printInitializationList(p, getBeforeArguments(), getInits(), " ");
  // p << " : ";
  // p.printFunctionalType(getInits().getTypes(), getResults().getTypes());
  // p << ' ';
  p.printRegion(getBefore(), /*printEntryBlockArgs=*/false);
  p << " do ";
  p.printRegion(getAfter());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs());
}

}
