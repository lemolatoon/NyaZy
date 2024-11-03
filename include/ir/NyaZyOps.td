#ifndef NYAZY_OPS
#define NYAZY_OPS

include "NyaZyDialect.td"
include "NyaZyBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

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
// CastOp
//===----------------------------------------------------------------------===//
def CastOp : NyaZyOp<"cast", 
    [Pure,
     DeclareOpInterfaceMethods<CastOpInterface>
     ]> {
  let summary = "type cast operation";
  let description = [{
    The "cast" operation converts an input value from one primitive type to an another primitive type. This corresponds to the NyaZy's "as $type" expression.
  }];
  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$out);
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
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
// MulOp
//===----------------------------------------------------------------------===//
def MulOp : NyaZyOp<"mul",
    [Pure]> {
  let summary = "multiplication operation";
  let description = [{
    The "nyazy.mul" operation represents the multiplication of two values.
  }];

  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//
def DivOp : NyaZyOp<"div",
    [Pure]> {
  let summary = "divide operation";
  let description = [{
    The "nyazy.div" operation represents the divide of two values.
  }];

  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);
}

//===----------------------------------------------------------------------===//
// PosOp
//===----------------------------------------------------------------------===//
def PosOp : NyaZyOp<"pos",
    [Pure]> {
  let summary = "unary positive operation";
  let description = [{
    The "nyazy.pos" operation represents the unary positive operation.
  }];

  let arguments = (ins I64:$lhs);
  let results = (outs I64);
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//
def NegOp : NyaZyOp<"neg",
    [Pure]> {
  let summary = "unary negative operation";
  let description = [{
    The "nyazy.pos" operation represents the unary negative operation.
  }];

  let arguments = (ins I64:$operand);
  let results = (outs I64);
}

def CmpOp
  : NyaZyOp<"cmp",
    [Pure]> {
  let summary = "comparison operation";
  let description = [{
    The `cmp` operation is a generic comparison for any types in nyazy. Its two
    arguments can be any types as long as their types
    match. The operation produces an i1 for the all cases.

    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   less than (mnemonic: `"lt"`; integer value: `2`)
    -   less than or equal (mnemonic: `"le"`; integer value: `3`)
    -   greater than (mnemonic: `"gt"`; integer value: `4`)
    -   greater than or equal (mnemonic: `"ge"`; integer value: `5`)

    The result is `1` if the comparison is true and `0` otherwise.

    Note: while the custom assembly form uses strings, the actual underlying
    attribute has integer type (or rather enum class in C++ code) as seen from
    the generic assembly form. String literals are used to improve readability
    of the IR by humans.
  }];

  let arguments = (ins NyaZy_CmpPredicateAttr:$predicate,
                       AnyType:$lhs,
                       AnyType:$rhs);
  let results = (outs UI1);

  let extraClassDeclaration = [{
    static std::optional<nyacc::CmpPredicate> getPredicateByName(mlir::StringRef name);
  }];

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `vs` type($rhs)";
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
