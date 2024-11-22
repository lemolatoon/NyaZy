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
include "mlir/Interfaces/MemorySlotInterfaces.td"

// Almost taken from `arith.constant`

//===----------------------------------------------------------------------===//
// ConstanOp
//===----------------------------------------------------------------------===//
def ConstantOp : NyaZyOp<"constant", 
    [Pure
    //  AllTypesMatch<["value", "result"]>,
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

    let assemblyFormat = "attr-dict $value `:` type($result)";
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

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def AllocaOp : NyaZyOp<"alloca", []> {
  let summary = "stack memory allocation operation";
  let description = [{
    The `alloca` operation allocates memory on the stack, to be automatically
    released when control transfers back from the region of its closest
    surrounding operation with an
    [`AutomaticAllocationScope`](../Traits.md/#automaticallocationscope) trait.
    The amount of memory allocated is specified by its memref and additional
    operands. For example:

    ```mlir
    %0 = memref.alloca() : memref<8x64xf32>
    ```

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type. In the example below, the SSA value '%d' is
    bound to the second dimension of the memref (which is dynamic).

    ```mlir
    %0 = memref.alloca(%d) : memref<8x?xf32>
    ```

    The optional list of symbol operands are bound to the symbols of the
    memref's affine map. In the example below, the SSA value '%s' is bound to
    the symbol 's0' in the affine map specified in the allocs memref type.

    ```mlir
    %0 = memref.alloca()[%s] : memref<8x64xf32,
                               affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>>
    ```

    This operation returns a single SSA value of memref type, which can be used
    by subsequent load and store operations. An optional alignment attribute, if
    specified, guarantees alignment at least to that boundary. If not specified,
    an alignment on any convenient boundary compatible with the type will be
    chosen.
  }];


  let arguments = (ins);
  let results = (outs AnyMemRef:$memref);
}

def StoreOp : NyaZyOp<"store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "::llvm::cast<mlir::MemRefType>($_self).getElementType()">,]> {
  let summary = "store operation";
  let description = [{
    Store a value to a memref location given by indices. The value stored should
    have the same type as the elemental type of the memref. The number of
    arguments provided within brackets need to match the rank of the memref.

    In an affine context, the indices of a store are restricted to SSA values
    bound to surrounding loop induction variables,
    [symbols](Affine.md/#restrictions-on-dimensions-and-symbols), results of a
    `constant` operation, or the result of an
    [`affine.apply`](Affine.md/#affineapply-affineapplyop) operation that can in
    turn take as arguments all of the aforementioned SSA values or the
    recursively result of such an `affine.apply` operation.

    Example:

    ```mlir
    memref.store %100, %A[%1, 1023] : memref<4x?xf32, #layout, memspace0>
    ```

    **Context:** The `load` and `store` operations are specifically crafted to
    fully resolve a reference to an element of a memref, and (in polyhedral
    `affine.if` and `affine.for` operations) the compiler can follow use-def
    chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop)
    operations) to precisely analyze references at compile-time using polyhedral
    techniques. This is possible because of the
    [restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols)
    in these contexts.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref);

  // let builders = [
  //   OpBuilder<(ins "mlir::Value":$valueToStore, "mlir::Value":$memref), [{
  //     $_state.addOperands(valueToStore);
  //     $_state.addOperands(memref);
  //   }]>];

  let extraClassDeclaration = [{
      mlir::Value getValueToStore() { return getOperand(0); }

      mlir::Value getMemRef() { return getOperand(1); }
      void setMemRef(mlir::Value value) { setOperand(1, value); }
      mlir::MemRefType getMemRefType() {
        return ::llvm::cast<mlir::MemRefType>(getMemRef().getType());
      }
  }];
  let assemblyFormat = [{
    $value `,` $memref attr-dict `:` type($memref)
  }];
}


//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : NyaZyOp<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "::llvm::cast<mlir::MemRefType>($_self).getElementType()">,]> {
  let summary = "load operation";
  let description = [{
    The `load` op reads an element from a memref specified by an index list. The
    output of load is a new value with the same type as the elements of the
    memref. The arity of indices is the rank of the memref (i.e., if the memref
    loaded from is of rank 3, then 3 indices are required for the load following
    the memref identifier).

    In an `affine.if` or `affine.for` body, the indices of a load are restricted
    to SSA values bound to surrounding loop induction variables,
    [symbols](Affine.md/#dimensions-and-symbols), results of a
    constant operations, or the result of an
    `affine.apply` operation that can in turn take as arguments all of the
    aforementioned SSA values or the recursively result of such an
    `affine.apply` operation.

    Example:

    ```mlir
    %1 = affine.apply affine_map<(d0, d1) -> (3*d0)> (%i, %j)
    %2 = affine.apply affine_map<(d0, d1) -> (d1+1)> (%i, %j)
    %12 = memref.load %A[%1, %2] : memref<8x?xi32, #layout, memspace0>

    // Example of an indirect load (treated as non-affine)
    %3 = affine.apply affine_map<(d0) -> (2*d0 + 1)>(%12)
    %13 = memref.load %A[%3, %2] : memref<4x?xi32, #layout, memspace0>
    ```

    **Context:** The `load` and `store` operations are specifically crafted to
    fully resolve a reference to an element of a memref, and (in affine
    `affine.if` and `affine.for` operations) the compiler can follow use-def
    chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop)
    operations) to precisely analyze references at compile-time using polyhedral
    techniques. This is possible because of the
    [restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols)
    in these contexts.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref);
  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
    mlir::Value getMemRef() { return getOperand(); }
    void setMemRef(mlir::Value value) { setOperand(value); }
    mlir::MemRefType getMemRefType() {
      return ::llvm::cast<mlir::MemRefType>(getMemRef().getType());
    }
  }];

  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : NyaZyOp<"while", []> {
  let summary = "a generic 'while' loop";
  let description = [{
    This operation represents a generic "while"/"do-while" loop that keeps
    iterating as long as a condition is satisfied. There is no restriction on
    the complexity of the condition. It consists of two regions (with single
    block each): "before" region and "after" region. The names of regions
    indicates whether they execute before or after the condition check.
    Therefore, if the main loop payload is located in the "before" region, the
    operation is a "do-while" loop. Otherwise, it is a "while" loop.

    The "before" region terminates with a special operation, `scf.condition`,
    that accepts as its first operand an `i1` value indicating whether to
    proceed to the "after" region (value is `true`) or not. The two regions
    communicate by means of region arguments. Initially, the "before" region
    accepts as arguments the operands of the `scf.while` operation and uses them
    to evaluate the condition. It forwards the trailing, non-condition operands
    of the `scf.condition` terminator either to the "after" region if the
    control flow is transferred there or to results of the `scf.while` operation
    otherwise. The "after" region takes as arguments the values produced by the
    "before" region and uses `scf.yield` to supply new arguments for the
    "before" region, into which it transfers the control flow unconditionally.

    A simple "while" loop can be represented as follows.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "while" loop, this region computes the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Forward the argument (as result or "after" region argument).
      scf.condition(%condition) %arg1 : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // In a "while" loop, this region is the loop body.
      %next = call @payload(%arg2) : (f32) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %next : f32
    }
    ```

    A simple "do-while" loop can be represented by reducing the "after" block
    to a simple forwarder.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "do-while" loop, this region contains the loop body.
      %next = call @payload(%arg1) : (f32) -> f32

      // And also evaluates the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Loop through the "after" region.
      scf.condition(%condition) %next : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // Forwards the values back to "before" region unmodified.
      scf.yield %arg2 : f32
    }
    ```

    Note that the types of region arguments need not to match with each other.
    The op expects the operand types to match with argument types of the
    "before" region; the result types to match with the trailing operand types
    of the terminator of the "before" region, and with the argument types of the
    "after" region. The following scheme can be used to share the results of
    some operations executed in the "before" region with the "after" region,
    avoiding the need to recompute them.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> i64 {
      // One can perform some computations, e.g., necessary to evaluate the
      // condition, in the "before" region and forward their results to the
      // "after" region.
      %shared = call @shared_compute(%arg1) : (f32) -> i64

      // Evaluate the condition.
      %condition = call @evaluate_condition(%arg1, %shared) : (f32, i64) -> i1

      // Forward the result of the shared computation to the "after" region.
      // The types must match the arguments of the "after" region as well as
      // those of the `scf.while` results.
      scf.condition(%condition) %shared : i64

    } do {
    ^bb0(%arg2: i64) {
      // Use the partial result to compute the rest of the payload in the
      // "after" region.
      %res = call @payload(%arg2) : (i64) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %res : f32
    }
    ```

    The custom syntax for this operation is as follows.

    ```
    op ::= `scf.while` assignments `:` function-type region `do` region
           `attributes` attribute-dict
    initializer ::= /* empty */ | `(` assignment-list `)`
    assignment-list ::= assignment | assignment `,` assignment-list
    assignment ::= ssa-value `=` ssa-value
    ```
  }];

  let arguments = (ins );
  let results = (outs );
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);

  let extraClassDeclaration = [{
    mlir::scf::ConditionOp getConditionOp() {
      return mlir::cast<mlir::scf::ConditionOp>(getBeforeBody()->getTerminator());
    }

    mlir::scf::YieldOp getYieldOp() {
      return mlir::cast<mlir::scf::YieldOp>(getAfterBody()->getTerminator());
    }

    mlir::Block *getBeforeBody() { return &getBefore().front(); }
    mlir::Block *getAfterBody() { return &getAfter().front(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

def ConditionOp : NyaZyOp<"condition", [
  HasParent<"WhileOp">,
  Pure,
  Terminator
]> {
  let summary = "loop continuation condition";
  let description = [{
    This operation accepts the continuation (i.e., inverse of exit) condition
    of the `scf.while` construct. If its first argument is true, the "after"
    region of `scf.while` is executed, with the remaining arguments forwarded
    to the entry block of the region. Otherwise, the loop terminates.
  }];

  let arguments = (ins UI1:$condition);

  let assemblyFormat =
      [{ `(` $condition `)` attr-dict  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : NyaZyOp<"yield", [Pure, Terminator,
  HasParent<"WhileOp">]> {
  let summary = "loop yield and termination operation";
  let description = [{
    "scf.yield" yields an SSA value from the SCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "scf.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "scf.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict  }];
}

def PrintOp : NyaZyOp<"print", []> {
  let summary = "print operation";
  let description = [{
    The `print` operation prints the value of its operand to the standard output.
    The operand can be of any type.
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs I32:$result);

  let assemblyFormat = "`(` $operand attr-dict `:` type($operand) `)` `->` type($result)";
}

#endif // NYAZY_OPS
