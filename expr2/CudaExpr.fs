module CudaExpr


type NShapeT = int list
type NStrideT = int list

type StorageT = NShapeT * NStrideT


type LeafOpT =
    | DiagonalOne

type UnaryOpT =
    | Negate

type BinaryOpT =
    | Add

type Expr =
    | Leaf of StorageT * LeafOpT
    | Unary of StorageT * UnaryOpT * Expr * Expr
    | Binary of StorageT * BinaryOpT * Expr * Expr




let rec fromOpExpr (expr: Op.ExprT) =
    match expr with
    | Op.Leaf(op) ->
        

