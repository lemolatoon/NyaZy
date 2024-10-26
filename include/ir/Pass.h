#include <memory>

namespace mlir {
class Pass;
};

namespace nyacc {

std::unique_ptr<mlir::Pass> createNyaZyToLLVMPass();

} // nyacc