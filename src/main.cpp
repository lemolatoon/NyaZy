#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include <iostream>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include "ast.h"
#include "lexer.h"
#include "mlirGen.h"
#include "parser.h"

#include "ir/NyaZyDialect.h"
#include "ir/NyaZyOps.h"
#include "ir/Pass.h"

int main() {
  std::string src = R"(
  let a = (3 == 3) as i64;
  let b = a + 9 + 3;
  let a = a * 2;
  a * 3 * b
)";
  llvm::outs() << "Source code:\n";
  llvm::outs() << src;
  nyacc::Lexer lexer(src);
  llvm::outs() << "Tokens:\n";
  const auto tokens = lexer.tokenize();

  if (!tokens) {
    std::cout << "Error: " << tokens.error().error(src) << "\n";
    return 1;
  };

  for (const auto &token : *tokens) {
    std::cout << token << "\n";
  }
  nyacc::Parser parser{*tokens};
  auto moduleAstOpt = parser.parseModule();
  if (!moduleAstOpt) {
    std::cout << moduleAstOpt.error().error(src) << "\n";
    return 1;
  }
  auto moduleAst = *moduleAstOpt;
  llvm::outs() << "AST:\n";
  moduleAst.dump();

  mlir::MLIRContext context;
  context.getOrLoadDialect<nyacc::NyaZyDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = nyacc::MLIRGen::gen(context, moduleAst);
  llvm::outs() << "MLIR:\n";
  module->dump();

  if (mlir::failed(mlir::verify(*module))) {
    llvm::errs() << "Module verification failed.\n";
    return 1;
  }

  mlir::PassManager pm(&context);
  pm.addPass(nyacc::createNyaZyToLLVMPass());

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Failed to lower to LLVM IR\n";
    return 1;
  }

  llvm::outs() << "Lowered MLIR:\n";
  module->dump();

  if (mlir::failed(mlir::verify(*module))) {
    llvm::errs() << "Module verification failed.\n";
    return 1;
  }

  // Convet the MLIR module to LLVM IR
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return 1;
  }

  // ファイルに書き出す
  std::error_code EC;
  llvm::raw_fd_ostream outputFile("output.ll", EC,
                                  llvm::sys::fs::OpenFlags::OF_None);

  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message() << "\n";
    return 1;
  }
  llvmModule->print(outputFile, nullptr);

  llvm::outs() << "Generated LLVM IR:\n";
  llvmModule->print(llvm::outs(), nullptr);

  return 0;
}
