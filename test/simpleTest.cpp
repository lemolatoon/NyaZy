// test/simpleTest.cpp
#include "ir/NyaZyDialect.h"
#include "ir/Pass.h"
#include "lexer.h"
#include "mlir/Pass/Pass.h"
#include "mlirGen.h"
#include "parser.h"
#include "gtest/gtest.h"
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

int runIR(std::unique_ptr<llvm::Module> &module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  llvm::orc::ThreadSafeContext context(std::make_unique<llvm::LLVMContext>());

  auto jit = llvm::orc::LLJITBuilder().create();
  EXPECT_TRUE(!!jit) << "Error creating LLJIT: "
                     << llvm::toString(jit.takeError()) << "\n";

  // Convert the module to ThreadSafeModule and add it to JIT
  llvm::orc::ThreadSafeModule tsm(std::move(module), context);
  auto err = jit->get()->addIRModule(std::move(tsm));
  EXPECT_FALSE(err) << "Error adding module: " << llvm::toString(std::move(err))
                    << "\n";

  // Specify the entry point function name (e.g., "main")
  auto symbol = jit->get()->lookup("main");
  EXPECT_TRUE(!!symbol) << "Error looking up symbol: "
                        << llvm::toString(symbol.takeError()) << "\n";

  auto mainFunction = symbol->toPtr<int (*)()>();
  int status = mainFunction();

  return status;
}

int runNyaZy(std::string src) {
  nyacc::Lexer lexer(src);
  auto tokens = lexer.tokenize();

  EXPECT_TRUE(tokens) << "Error: " << tokens.error().error(src) << "\n";

  nyacc::Parser parser(*tokens);
  auto ast = parser.parseModule();

  mlir::MLIRContext context;
  context.getOrLoadDialect<nyacc::NyaZyDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = nyacc::MLIRGen::gen(context, ast);

  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)))
      << "Module verification failed:\n"
      << src << "\n";

  mlir::PassManager pm(&context);
  pm.addPass(nyacc::createNyaZyToLLVMPass());

  EXPECT_TRUE(mlir::succeeded(pm.run(*module))) << "PassManager failed:\n"
                                                << src << "\n";

  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  std::cerr << llvmModule << "\n";
  EXPECT_TRUE(llvmModule) << "Failed to emit LLVM IR:\n" << src << "\n";
  llvmModule->dump();

  return runIR(llvmModule);
};

// テストケース
TEST(SimpleTest, OneInteger) { EXPECT_EQ(123, runNyaZy("123")); }

TEST(SimpleTest, ArithOps) {
  EXPECT_EQ(3, runNyaZy("1+2"));
  EXPECT_EQ(8, runNyaZy("1+2+5"));
  EXPECT_EQ(4, runNyaZy("1*2+5/2"));
  EXPECT_EQ(3, runNyaZy("1*(2+5)/2"));
  EXPECT_EQ(3, runNyaZy("+2+1"));
  EXPECT_EQ(-1, runNyaZy("-2+1"));
}

TEST(SimpleTest, CompareOps) {
  EXPECT_EQ(1, runNyaZy("(3 == 3) as i64"));

  EXPECT_EQ(1, runNyaZy("(3 >= -3) as i64"));
  EXPECT_EQ(1, runNyaZy("(3 > -3) as i64"));

  EXPECT_EQ(0, runNyaZy("(3 <= -3) as i64"));
  EXPECT_EQ(0, runNyaZy("(3 < -3) as i64"));
}

// メイン関数
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
