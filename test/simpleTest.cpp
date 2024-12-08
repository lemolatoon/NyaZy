// test/simpleTest.cpp
#include "ir/NyaZyDialect.h"
#include "ir/Pass.h"
#include "lexer.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlirGen.h"
#include "parser.h"
#include "gtest/gtest.h"
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>>
createLLJIT(std::unique_ptr<llvm::Module> &module) {
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

  return jit;
}

int runIR(std::unique_ptr<llvm::Module> &module) {
  auto jit = createLLJIT(module);
  // Specify the entry point function name (e.g., "main")
  auto symbol = jit->get()->lookup("main");
  EXPECT_TRUE(!!symbol) << "Error looking up symbol: "
                        << llvm::toString(symbol.takeError()) << "\n";

  auto mainFunction = symbol->toPtr<int (*)()>();
  int status = mainFunction();

  return status;
}

std::string runIRWithCapturedOutput(std::unique_ptr<llvm::Module> &module) {
  auto jit = createLLJIT(module);
  // Specify the entry point function name (e.g., "main")
  auto symbol = jit->get()->lookup("main");
  EXPECT_TRUE(!!symbol) << "Error looking up symbol: "
                        << llvm::toString(symbol.takeError()) << "\n";

  auto mainFunction = symbol->toPtr<int (*)()>();

  // Create a pipe for capturing stdout
  int pipefd[2];
  EXPECT_NE(pipe(pipefd), -1)
      << "Error creating pipe: " << strerror(errno) << "\n";

  // Fork the process
  pid_t pid = fork();
  EXPECT_NE(pid, -1) << "Error forking process: " << strerror(errno) << "\n";

  if (pid == 0) {
    // Child process
    close(pipefd[0]);               // Close read end of the pipe
    dup2(pipefd[1], STDOUT_FILENO); // Redirect stdout to the pipe
    close(pipefd[1]);               // Close write end of the pipe

    // Execute the main function
    int status = mainFunction();
    fflush(stdout); // Flush stdout
    _exit(status);  // Exit the child process
  } else {
    // Parent process
    close(pipefd[1]); // Close write end of the pipe

    // Read the output from the pipe
    std::stringstream buffer;
    char readBuffer[128];
    ssize_t bytesRead;
    while ((bytesRead = read(pipefd[0], readBuffer, sizeof(readBuffer) - 1)) >
           0) {
      readBuffer[bytesRead] = '\0';
      buffer << readBuffer;
    }
    close(pipefd[0]); // Close read end of the pipe

    // Wait for the child process to finish
    int status;
    waitpid(pid, &status, 0);

    // Return the captured output
    return buffer.str();
  }
  return "";
}

std::unique_ptr<llvm::Module> createLLVMModule(const std::string src,
                                               llvm::LLVMContext &llvmContext) {
  nyacc::Lexer lexer(src);
  auto tokens = lexer.tokenize();

  EXPECT_TRUE(tokens) << "Error: " << tokens.error().error(src) << "\n";

  nyacc::Parser parser(*tokens);
  auto ast = parser.parseModule();
  EXPECT_TRUE(ast) << ast.error().error(src) << "\n";

  mlir::MLIRContext context;
  context.getOrLoadDialect<nyacc::NyaZyDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  auto moduleOpt = nyacc::MLIRGen::gen(context, *ast);
  EXPECT_TRUE(moduleOpt) << moduleOpt.error().error(src) << "\n";
  auto &module = *moduleOpt;

  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)))
      << "Module verification failed:\n"
      << src << "\n";

  context.disableMultithreading();

  mlir::PassManager pm(&context);
  pm.addPass(nyacc::createNyaZyToStdPass());
  pm.addPass(nyacc::createStdToLLVMPass());
  pm.addPass(nyacc::createNyaZyPrintToLLVMPass());

  const auto shouldPrintBefore =
      std::function{[](mlir::Pass *, mlir::Operation *) { return false; }};
  const auto shouldPrintAfter =
      std::function{[](mlir::Pass *, mlir::Operation *) { return true; }};
  pm.enableIRPrinting(shouldPrintBefore, shouldPrintAfter, true, false, true,
                      llvm::outs(), mlir::OpPrintingFlags{});

  context.enableMultithreading();

  EXPECT_TRUE(mlir::succeeded(pm.run(*module))) << "PassManager failed:\n"
                                                << src << "\n";

  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  EXPECT_TRUE(llvmModule) << "Failed to emit LLVM IR:\n" << src << "\n";

  return llvmModule;
}

int runNyaZy(std::string src) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = createLLVMModule(src, llvmContext);
  return runIR(llvmModule);
};

std::string runNyaZyWithCapturedOutput(std::string src) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = createLLVMModule(src, llvmContext);
  return runIRWithCapturedOutput(llvmModule);
}

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

TEST(SimpleTest, Variable) {
  EXPECT_EQ(10, runNyaZy("let a = 5 * 2; a"));

  EXPECT_EQ(8, runNyaZy("let a = 5; let b = 3; a + b"));

  // assign
  EXPECT_EQ(7, runNyaZy("let a = 5; a = 7; a"));
  // assign using self
  EXPECT_EQ(12, runNyaZy("let a = 5; a = a + 7; a"));
  // shadowing
  EXPECT_EQ(12, runNyaZy("let a = 5; let a = a + 7; a"));
}

TEST(SimpleTest, While) {
  // block expr
  EXPECT_EQ(13, runNyaZy("{ let a = 5; a + 8 }"));
  // block expr's stmt
  EXPECT_EQ(7, runNyaZy("let a = 5; { a = 7; }; a"));
  EXPECT_EQ(10, runNyaZy("let a = 0; while (a < 10) { a = a + 1; } a"));
  EXPECT_EQ(55, runNyaZy(R"(
        let a = 0;
        let sum = 0;
        while (a <= 10) {
            sum = sum + a;
            a = a + 1;
        }
        sum
      )"));
}

TEST(SimpleTest, PrintOp) {
  EXPECT_EQ("124\n", runNyaZyWithCapturedOutput("print(123 + 1); 0"));
  EXPECT_EQ("123\n456\n",
            runNyaZyWithCapturedOutput("print(123); print(456); 0"));
  EXPECT_EQ("123\n456\n",
            runNyaZyWithCapturedOutput("print(123); print(456); 0"));
  EXPECT_EQ("Hello World!\n",
            runNyaZyWithCapturedOutput("print(\"Hello World!\"); 0"));
}

// メイン関数
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
