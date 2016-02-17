namespace SymTensor

open System.Collections.Generic
open Expr

[<AutoOpen>]
module UExprTypes = 

    type IUExtensionOp =
        abstract Arity: ArityT with get        

    type ULeafOpT =
        | Identity of SizeSpecT
        | Zeros of ShapeSpecT                   
        | ScalarConst of obj
        | Var of IVarSpec

    type UUnaryOpT =
        | Negate                        
        | Log                           
        | Exp                           
        | Sum                           
        | SumAxis of int                
        | Reshape of ShapeSpecT         
        | Broadcast of ShapeSpecT       
        | SwapDim of int * int          
        | StoreToVar of IVarSpec
        | Annotated of Annotation       

    type UBinaryOpT =
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Power                         
        | Dot                           
        | TensorProduct                 

    type UNaryOpT =
        | Discard        
        | ExtensionOp of IUExtensionOp
             

    /// unified op of any arity and type
    type UOpT =
        | ULeafOp of ULeafOpT
        | UUnaryOp of UUnaryOpT
        | UBinaryOp of UBinaryOpT
        | UNaryOp of UNaryOpT

    /// unified expression (combines all arities and types)
    type UExprT = UExpr of UOpT * System.Type * (UExprT list)


module UExpr =

    /// extracts the top-level op from an expression
    let inline extractOp (expr: ExprT<'T>) =
        match expr with
        | Leaf op -> 
            match op with
            | Expr.Identity ss -> Identity ss
            | Expr.Zeros ss -> Zeros ss
            | Expr.ScalarConst v -> ScalarConst (box v)
            | Expr.Var vs -> Var (vs :> IVarSpec)
            |> ULeafOp
        | Unary(op, _) -> 
            match op with
            | Expr.Negate -> Negate
            | Expr.Log -> Log
            | Expr.Exp -> Exp                          
            | Expr.Sum -> Sum                           
            | Expr.SumAxis a -> SumAxis a            
            | Expr.Reshape ss -> Reshape ss
            | Expr.Broadcast ss -> Broadcast ss
            | Expr.SwapDim (ax1, ax2) -> SwapDim (ax1, ax2)
            | Expr.StoreToVar vs -> StoreToVar (vs :> IVarSpec)
            | Expr.Annotated ano -> Annotated ano
            |> UUnaryOp
        | Binary(op, _, _) -> 
            match op with
            | Expr.Add -> Add                         
            | Expr.Substract -> Substract                    
            | Expr.Multiply -> Multiply                     
            | Expr.Divide -> Divide                       
            | Expr.Power -> Power               
            | Expr.Dot -> Dot                   
            | Expr.TensorProduct -> TensorProduct     
            |> UBinaryOp
        | Nary(op, _) -> 
            match op with
            | Expr.Discard -> Discard
            | Expr.ExtensionOp eop -> ExtensionOp (eop :?> IUExtensionOp)
            |> UNaryOp

    /// converts an expression to a unified expression
    let rec toUExpr (expr: ExprT<'T>) =
        let uop = extractOp expr
        match expr with
        | Leaf op -> UExpr (uop, typeof<'T>, [])
        | Unary(op, a) -> UExpr(uop, typeof<'T>, [toUExpr a])
        | Binary(op, a, b) -> UExpr(uop, typeof<'T>, [toUExpr a; toUExpr b])
        | Nary(op, se) -> UExpr(uop, typeof<'T>, se |> List.map toUExpr)
    
    /// converts a unified expression to an expression
//    let rec toExpr uexpr =
//        match uexpr with
//        | UExpr(LeafOp op, []) -> Leaf op
//        | UExpr(UnaryOp op, [a]) -> Unary(op, toExpr a)
//        | UExpr(BinaryOp op, [a; b]) -> Binary(op, toExpr a, toExpr b)
//        | UExpr(NaryOp op, se) -> Nary(op, se |> List.map toExpr)
//        | _ -> failwithf "invalid unified expression %A" uexpr

    /// Return the shape of the given unified expression.
    //let shapeOf uexpr = uexpr |> toExpr |> shapeOf

    /// counts how many times subExpr occurs in unified expression uexpr
    let subExprOccurrences uexpr =
        let cnt = Dictionary<UExprT, int>()
        let rec build expr =
            if cnt.ContainsKey(expr) then
                cnt.[expr] <- cnt.[expr] + 1
            else
                cnt.[expr] <- 1

            match expr with
            | UExpr (_, _, srcs) ->
                for src in srcs do
                    build src
        build uexpr

        fun subExpr ->
            if cnt.ContainsKey(subExpr) then cnt.[subExpr]
            else 0

