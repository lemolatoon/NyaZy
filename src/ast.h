#pragma once

#include <cstdint>

namespace nyacc {
    class Visitor {
        public:
            virtual ~Visitor() = default;
            virtual void visit(class NumLitExpr &node) = 0;
    };

    class ExprASTNode {
        public:
            enum class ExprKind {
                NumLit,
            };
            ExprASTNode(ExprKind kind) : kind_(kind) {}
            virtual ~ExprASTNode() = default;
            virtual void accept(class Visitor &v) = 0;
            ExprKind getKind() const {
                return kind_;
            };
        private:
            ExprKind kind_;
    };


    class NumLitExpr : public ExprASTNode {
        public:
            NumLitExpr(int64_t value) : ExprASTNode(ExprKind::NumLit), value_(value) {}

            void accept(Visitor &v) override;
            int64_t getValue() const {
                return value_;
            }

            static bool classof(const ExprASTNode *node) {
                return node->getKind() == ExprKind::NumLit;
            }
        private:
            int64_t value_;
    };
}