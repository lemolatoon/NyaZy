#include "mlirGen.h"
#include "ir/NyaZyOps.h"
#include "ast.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace {

class MLIRGenVisitor : public nyacc::Visitor {
public:
    MLIRGenVisitor(mlir::MLIRContext &context)
        : builder_(&context), module_(mlir::ModuleOp::create(builder_.getUnknownLoc())), value_(std::nullopt) {
        builder_.setInsertionPointToStart(module_.getBody());
        }
    
    mlir::OwningOpRef<mlir::ModuleOp> takeModule() {
        return std::move(module_);
    }
    
    void visit(const nyacc::ModuleAST &moduleAst) override {
        moduleAst.getExpr()->accept(*this);
    }

    void visit(const nyacc::NumLitExpr &numLit) override {
        value_ = builder_.create<nyacc::ConstantOp>(
            builder_.getUnknownLoc(), builder_.getI64IntegerAttr(numLit.getValue()));
    }

private:
    mlir::OpBuilder builder_;
    mlir::ModuleOp module_;
    std::optional<mlir::Value> value_;
};

}

namespace nyacc {

mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::gen(mlir::MLIRContext &context, const ModuleAST &moduleAst) {
    MLIRGenVisitor visitor{context};
    moduleAst.accept(visitor);
    return visitor.takeModule();
}

} // nyacc