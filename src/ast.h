#pragma once

#include <cstdint>
#include <memory>

namespace nyacc {
    class Visitor {
        public:
            virtual ~Visitor() = default;
            virtual void visit(class ModuleAST &node) = 0;
            virtual void visit(class NumLitExpr &node) = 0;
    };


    class ExprASTNode {
        public:
            enum class ExprKind {
                NumLit,
            };
            explicit ExprASTNode(ExprKind kind) : kind_(kind) {}
            virtual ~ExprASTNode() = default;
            virtual void accept(class Visitor &v) = 0;
            virtual void dump(int level) const = 0;
            ExprKind getKind() const {
                return kind_;
            };
        private:
            ExprKind kind_;
    };

    class NumLitExpr : public ExprASTNode {
        public:
            NumLitExpr(int64_t value) : ExprASTNode(ExprKind::NumLit), value_(value) {}

            void accept(Visitor &v) override {
                v.visit(*this);
            }
            int64_t getValue() const {
                return value_;
            }

            static bool classof(const ExprASTNode *node) {
                return node->getKind() == ExprKind::NumLit;
            }

            void dump(int level) const override;
        private:
            int64_t value_;
    };

    class ModuleAST {
        public:
            ModuleAST(std::unique_ptr<ExprASTNode> expr) : expr_(std::move(expr)) {}
            void accept(Visitor &v) {
                v.visit(*this);
            };
            void dump(int level = 0) const;
        private:
            std::unique_ptr<ExprASTNode> expr_;
    };
}