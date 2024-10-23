#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinDialect.h"
#include <iostream>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include "lexer.h"
#include "ast.h"

mlir::LLVM::LLVMFunctionType
  getPrintfType(mlir::MLIRContext *context) {
    auto llvmI32Type = mlir::IntegerType::get(context, 32);
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(context);
    auto llvmPrintfType = mlir::LLVM::LLVMFunctionType::get(
        llvmI32Type, llvmPtrType, /*isVarArg=*/true);
    return llvmPrintfType;
}

mlir::FlatSymbolRefAttr getOrInsertPrintf(mlir::ModuleOp module) {
    auto* context = module.getContext();
    const char *printfSymbol = "printf";

    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(printfSymbol)) {
        return mlir::SymbolRefAttr::get(context, printfSymbol);
    }

    auto llvmPrintfType = getPrintfType(context);

    mlir::PatternRewriter rewriter{context};
    mlir::PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(
        module.getLoc(), printfSymbol, llvmPrintfType);
    
    return mlir::SymbolRefAttr::get(context, printfSymbol);
} 

mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder &builder, mlir::StringRef name, mlir::StringRef value, mlir::ModuleOp module) {
    mlir::LLVM::GlobalOp global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name);
    if (!global) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto type = mlir::LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(builder.getContext(), 8), value.size()
        );
        global = builder.create<mlir::LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
            builder.getStringAttr(value), /*alignment=*/0
        );
    }

    // Get the pointer to the first char in the global string.
    mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(
        loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getIndexAttr(0));
    
    // get element pointer
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto gep = builder.create<mlir::LLVM::GEPOp>(
        loc, /*resultType=*/llvmPtrType, /*elementType=*/global.getType(), /*basePtr=*/globalPtr,
        mlir::ArrayRef<mlir::Value>({/*base addr=*/cst0, /*index=*/cst0})
    );
    return gep;
}

int main() {

    nyacc::Lexer lexer("123");
    const auto tokens = lexer.tokenize();
    for (const auto &token : tokens) {
        std::cout << token << "\n";
    }

    // Initialize MLIR context
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::BuiltinDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    // Create an empty module
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    builder.setInsertionPointToStart(module.getBody());
    auto mainOp = builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "main",
        mlir::LLVM::LLVMFunctionType::get(builder.getI32Type(), {}, false));
    
    auto entryBlock = mainOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    auto printfRef = getOrInsertPrintf(module);
    auto printfType = getPrintfType(&context);
    const auto helloWorldPtrValue = getOrCreateGlobalString(
        builder.getUnknownLoc(), builder, "hello_world", "Hello, World!\n", module);
    builder.create<mlir::LLVM::CallOp>(
        builder.getUnknownLoc(), printfType, printfRef,
        helloWorldPtrValue
    );

    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32Type(), builder.getIndexAttr(0));
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), cst0);

    // Verify the module to ensure everything is valid
    if (failed(mlir::verify(module))) {
        llvm::errs() << "Module verification failed.\n";
        return 1;
    }

    llvm::outs() << "Generated MLIR:\n";
    // Print the generated MLIR module
    module.print(llvm::outs());
    llvm::outs() << "\n";

    // Convet the MLIR module to LLVM IR
    mlir::registerBuiltinDialectTranslation(*module.getContext());
    mlir::registerLLVMDialectTranslation(*module.getContext());
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return 1;
    }
    llvm::outs() << "Generated LLVM IR:\n";
    llvmModule->print(llvm::outs(), nullptr);

    return 0;
}
