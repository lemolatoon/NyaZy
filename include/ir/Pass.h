#include <memory>

namespace mlir {
class Pass;
};

namespace nyacc {

// Lower NyaZy dialect to standard dialects: arith, memref, scf, func, except for `nyazy.print`
std::unique_ptr<mlir::Pass> createNyaZyToStdPass();
// Lower standard dialects(arith, memref, scf, func) to LLVM dialect
std::unique_ptr<mlir::Pass> createStdToLLVMPass();
// Lower `nyazy.print` to LLVM printf
std::unique_ptr<mlir::Pass> createNyaZyPrintToLLVMPass();

} // nyacc