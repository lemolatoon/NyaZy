#ifndef NYAZY_DIALECT
#define NYAZY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// NyaZy dialect definition.
//===----------------------------------------------------------------------===//

def NyaZyDialect : Dialect {
    let name = "nyazy";
    let summary = "NyaZy language dialect.";
    let description = [{
        This dialect is NyaZy language dialect aimed to one to one mapping to the NyaZy language AST.
    }];
    let cppNamespace = "nyacc";
}

//===----------------------------------------------------------------------===//
// Base NyaZy operation definition.
//===----------------------------------------------------------------------===//

class NyaZyOp<string mnemonic, list<Trait> traits = []> :
        Op<NyaZyDialect, mnemonic, traits>;

#endif // NYAZY_DIALECT
