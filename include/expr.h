#include <memory>
namespace nyacc {
class ExprASTNode;
class StmtASTNode;
using Expr = std::shared_ptr<ExprASTNode>;
using Stmt = std::shared_ptr<StmtASTNode>;
} // namespace nyacc
