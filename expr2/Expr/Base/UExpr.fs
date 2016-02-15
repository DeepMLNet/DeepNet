namespace ExprNS

open System.Collections.Generic
open Expr

module UExpr =

    /// an op of any arity
    type AnyOpT<'T> =
        | LeafOp of LeafOpT<'T>
        | UnaryOp of UnaryOpT<'T>
        | BinaryOp of BinaryOpT<'T>
        | NaryOp of NaryOpT<'T>

    /// unified expression (combines all arities into one type)
    type UExprT<'T> = UExpr of AnyOpT<'T> * (UExprT<'T> list)

    /// extracts the top-level op from an expression
    let extractOp expr =
        match expr with
        | Leaf op -> LeafOp op
        | Unary(op, _) -> UnaryOp op
        | Binary(op, _, _) -> BinaryOp op
        | Nary(op, _) -> NaryOp op

    /// converts an expression to a unified expression
    let rec toUExpr expr =
        match expr with
        | Leaf op -> UExpr(LeafOp op, [])
        | Unary(op, a) -> UExpr(UnaryOp op, [toUExpr a])
        | Binary(op, a, b) -> UExpr(BinaryOp op, [toUExpr a; toUExpr b])
        | Nary(op, se) -> UExpr(NaryOp op, se |> List.map toUExpr)
    
    /// converts a unified expression to an expression
    let rec toExpr uexpr =
        match uexpr with
        | UExpr(LeafOp op, []) -> Leaf op
        | UExpr(UnaryOp op, [a]) -> Unary(op, toExpr a)
        | UExpr(BinaryOp op, [a; b]) -> Binary(op, toExpr a, toExpr b)
        | UExpr(NaryOp op, se) -> Nary(op, se |> List.map toExpr)
        | _ -> failwithf "invalid unified expression %A" uexpr

    /// Return the shape of the given unified expression.
    let shapeOf uexpr = uexpr |> toExpr |> shapeOf

    /// counts how many times subExpr occurs in unified expression uexpr
    let subExprOccurrences uexpr =
        let cnt = Dictionary<UExprT<'T>, int>()
        let rec build expr =
            if cnt.ContainsKey(expr) then
                cnt.[expr] <- cnt.[expr] + 1
            else
                cnt.[expr] <- 1

            match expr with
            | UExpr (_, srcs) ->
                for src in srcs do
                    build src
        build uexpr

        fun subExpr ->
            if cnt.ContainsKey(subExpr) then cnt.[subExpr]
            else 0

