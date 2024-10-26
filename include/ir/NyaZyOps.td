#ifndef NYAZY_OPS
#define NYAZY_OPS

include "NyaZyDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Almost taken from `arith.constant`

//===----------------------------------------------------------------------===//
// ConstanOp
//===----------------------------------------------------------------------===//
def ConstantOp : NyaZyOp<"constant", 
    [Pure,
     AllTypesMatch<["value", "result"]>,
     ]> {
    let summary = "integer or floating point constant operation";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute.

        TODO: Example:

        ```mlir
        %0 = "nyazy.constant" 2 : i32
        // Equivalent generic form
        %1 = "nyazy.constant"() {value = 42 : i32} : () -> i32
        ```
    }];

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs /*SignlessIntegerOrFloatLike*/AnyType:$result);

    let assemblyFormat = "attr-dict $value";
}

//===----------------------------------------------------------------------===//
// AddOp
// reference: thirdparty/build/llvm/src/llvm_project/mlir/examples/toy/Ch7/include/toy/Ops.td
//===----------------------------------------------------------------------===//
def AddOp : NyaZyOp<"add",
    [Pure]> {
  let summary = "addition operation";
  let description = [{
    The "nyazy.add" operation represents the addition of two values.
  }];

  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);

  // Allow building an AddOp with from the two input operands.
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//
def SubOp : NyaZyOp<"sub",
    [Pure]> {
  let summary = "subtraction operation";
  let description = [{
    The "nyazy.sub" operation represents the subtraction of two values.
  }];

  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//
def FuncOp : NyaZyOp<"func", [
    FunctionOpInterface,
    IsolatedFromAbove,
]> {
    let summary = "function operation";
    let description = [{
        The "nyazy.func" operation represents a function in the NyaZy language.
        Currently the main function is implicitly defined in the module.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilder<(ins
            "mlir::StringRef":$name, "mlir::FunctionType":$type,
            CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs
        )>
    ];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        mlir::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        mlir::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

        mlir::Region *getCallableRegion() { return &getBody(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def ReturnOp : NyaZyOp<"return", 
    [Terminator]> {
    let summary = "return operation";
    let description = [{
        Return operation terminates the program with a given status code.
        This operation is temporary added to this dialect to support the `exiting with the expression result as status code`.
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs);
}

#endif // NYAZY_OPS
