#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include <fstream>
#include <iostream>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
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

std::string readFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Could not open file: " + filename);
  }
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  return content;
}

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputFileName(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);
  llvm::cl::opt<std::string> outputFileName(
      "o", llvm::cl::desc("Specify output file"),
      llvm::cl::value_desc("filename"));
  llvm::cl::opt<bool> debugOutput("d", llvm::cl::desc("Enable debug output"),
                                  llvm::cl::init(false));

  llvm::cl::ParseCommandLineOptions(argc, argv, "nyazy compiler\n");

  std::string src;
  try {
    src = readFile(inputFileName);
  } catch (const std::exception &e) {
    llvm::errs() << e.what() << "\n";
    return 1;
  }

  if (debugOutput) {
    llvm::outs() << "Source code:\n";
    llvm::outs() << src;
  }
  nyacc::Lexer lexer(src, std::make_shared<std::string>(inputFileName));
  if (debugOutput) {
    llvm::outs() << "Tokens:\n";
  }
  const auto tokens = lexer.tokenize();

  if (!tokens) {
    std::cout << "Error: " << tokens.error().error(src) << "\n";
    return 1;
  };

  if (debugOutput) {
    for (const auto &token : *tokens) {
      std::cout << token << "\n";
    }
  }
  nyacc::Parser parser{*tokens};
  auto moduleAstOpt = parser.parseModule();
  if (!moduleAstOpt) {
    std::cout << moduleAstOpt.error().error(src) << "\n";
    return 1;
  }
  auto moduleAst = *moduleAstOpt;
  if (debugOutput) {
    llvm::outs() << "AST:\n";
    moduleAst.dump();
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<nyacc::NyaZyDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  auto moduleOpt = nyacc::MLIRGen::gen(context, moduleAst);
  if (!moduleOpt) {
    std::cout << moduleOpt.error().error(src) << "\n";
    return 1;
  }
  auto &module = *moduleOpt;
  if (debugOutput) {
    llvm::outs() << "MLIR:\n";
    module->dump();
  }

  if (mlir::failed(mlir::verify(*module))) {
    llvm::errs() << "Module verification failed.\n";
    return 1;
  }

  mlir::PassManager pm(&context);
  pm.addPass(nyacc::createNyaZyToStdPass());
  pm.addPass(nyacc::createStdToLLVMPass());
  pm.addPass(nyacc::createNyaZyPrintToLLVMPass());

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Failed to lower to LLVM IR\n";
    return 1;
  }

  if (debugOutput) {
    llvm::outs() << "Lowered MLIR:\n";
    module->dump();
  }

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

  std::string outputFilename =
      outputFileName.empty()
          ? inputFileName.substr(0, inputFileName.find_last_of('.')) + ".ll"
          : outputFileName;

  // ファイルに書き出す
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(outputFilename, EC,
                                  llvm::sys::fs::OpenFlags::OF_None);

  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message() << "\n";
    return 1;
  }
  if (debugOutput) {
    llvm::outs() << "Generated LLVM IR:\n";
    llvmModule->print(llvm::outs(), nullptr);
  }

  llvmModule->print(outputFile, nullptr);

  return 0;
}
