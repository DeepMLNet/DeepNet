namespace SymTensor

open System.Reflection
open System.Collections.Generic
open Expr

[<AutoOpen>]
module UExprTypes = 


    type IUExtensionOp =
        inherit System.IComparable
        abstract Arity: ArityT with get        

    type ULeafOpT =
        | Identity of SizeSpecT
        | Zeros of ShapeSpecT                   
        | ScalarConst of System.IComparable
        | Var of IVarSpec

    type UUnaryOpT =
        | Negate                        
        | Abs
        | SignT
        | Log
        | Log10                           
        | Exp                           
        | Sin
        | Cos
        | Tan
        | Asin
        | Acos
        | Atan
        | Sinh
        | Cosh
        | Tanh
        | Sqrt
        | Ceil
        | Floor
        | Round
        | Truncate                   
        | Sum                           
        | SumAxis of int                
        | Reshape of ShapeSpecT         
        | DoBroadcast of ShapeSpecT       
        | SwapDim of int * int          
        | StoreToVar of IVarSpec
        | Annotated of Annotation       

    type UBinaryOpT =
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide       
        | Modulo                 
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
    [<StructuralComparison; StructuralEquality; StructuredFormatDisplay("{PrettyString}")>]
    type UExprT = 
        | UExpr of UOpT * TypeNameT * ShapeSpecT * (UExprT list)

        member this.PrettyString =
            match this with
            | UExpr (ULeafOp uop, tn, ss, subs) -> sprintf "%A" uop 
            | UExpr (UUnaryOp uop, tn, ss, subs) -> sprintf "%A (%A)" uop subs.[0]
            | UExpr (UBinaryOp uop, tn, ss, subs) -> sprintf "%A (%A, %A)" uop subs.[0] subs.[1]
            | UExpr (UNaryOp uop, tn, ss, subs) -> sprintf "%A (%A)" uop subs


module UExpr =

    /// extracts the top-level op from an expression
    let inline extractOp (expr: ExprT<'T>) =
        match expr with
        | Leaf op -> 
            match op with
            | Expr.Identity ss -> Identity ss
            | Expr.Zeros ss -> Zeros ss
            | Expr.ScalarConst v -> ScalarConst (box v :?> System.IComparable)
            | Expr.Var vs -> Var (vs :> IVarSpec)
            |> ULeafOp
        | Unary(op, _) -> 
            match op with
            | Expr.Negate -> Negate
            | Expr.Abs -> Abs
            | Expr.SignT -> SignT
            | Expr.Log -> Log
            | Expr.Log10 -> Log10
            | Expr.Exp -> Exp                          
            | Expr.Sin -> Sin
            | Expr.Cos -> Cos
            | Expr.Tan -> Tan
            | Expr.Asin -> Asin
            | Expr.Acos -> Acos
            | Expr.Atan -> Atan
            | Expr.Sinh -> Sinh
            | Expr.Cosh -> Cosh
            | Expr.Tanh -> Tanh
            | Expr.Sqrt -> Sqrt
            | Expr.Ceil -> Ceil
            | Expr.Floor -> Floor
            | Expr.Round -> Round
            | Expr.Truncate -> Truncate
            | Expr.Sum -> Sum                           
            | Expr.SumAxis a -> SumAxis a            
            | Expr.Reshape ss -> Reshape ss
            | Expr.DoBroadcast ss -> DoBroadcast ss
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
            | Expr.Modulo -> Modulo          
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
        let tn = TypeName typeof<'T>.AssemblyQualifiedName
        let shp = Expr.shapeOf expr
        match expr with
        | Leaf op -> UExpr (uop, tn, shp, [])
        | Unary(op, a) -> UExpr(uop, tn, shp, [toUExpr a])
        | Binary(op, a, b) -> UExpr(uop, tn, shp, [toUExpr a; toUExpr b])
        | Nary(op, se) -> UExpr(uop, tn, shp, se |> List.map toUExpr)
    
    /// converts a unified expression to an expression of (known) type
    let rec toExprOfType (UExpr (uop, tn, ss, subUExprs) as uexpr) : ExprT<'T> =
        if TypeName.ofType<'T> <> tn then
            failwith "UExpr type does not match does function"

        match uop with
        | ULeafOp uop ->
            match uop with
            | Identity ss -> Expr.Identity ss
            | Zeros ss -> Expr.Zeros ss
            | ScalarConst v -> Expr.ScalarConst (box v :?> 'T)
            | Var vs -> Expr.Var (box vs :?> VarSpecT<'T>)
            |> Expr.Leaf
        | UUnaryOp uop ->
            match uop with
            | Negate -> Expr.Negate
            | Abs -> Expr.Abs
            | SignT -> Expr.SignT
            | Log -> Expr.Log
            | Log10 -> Expr.Log10
            | Exp -> Expr.Exp                          
            | Sin -> Expr.Sin
            | Cos -> Expr.Cos
            | Tan -> Expr.Tan
            | Asin -> Expr.Asin
            | Acos -> Expr.Acos
            | Atan -> Expr.Atan
            | Sinh -> Expr.Sinh
            | Cosh -> Expr.Cosh
            | Tanh -> Expr.Tanh
            | Sqrt -> Expr.Sqrt
            | Ceil -> Expr.Ceil
            | Floor -> Expr.Floor
            | Round -> Expr.Round
            | Truncate -> Expr.Truncate
            | Sum -> Expr.Sum                           
            | SumAxis a -> Expr.SumAxis a            
            | Reshape ss -> Expr.Reshape ss
            | DoBroadcast ss -> Expr.DoBroadcast ss
            | SwapDim (ax1, ax2) -> Expr.SwapDim (ax1, ax2)
            | StoreToVar vs -> Expr.StoreToVar (box vs :?> VarSpecT<'T>)
            | Annotated ano -> Expr.Annotated ano
            |> fun op -> Expr.Unary (op, toExprOfType subUExprs.[0])
        | UBinaryOp uop ->
            match uop with
            | Add -> Expr.Add                         
            | Substract -> Expr.Substract                    
            | Multiply -> Expr.Multiply                     
            | Divide -> Expr.Divide             
            | Modulo -> Expr.Modulo          
            | Power -> Expr.Power               
            | Dot -> Expr.Dot                   
            | TensorProduct -> Expr.TensorProduct     
            |> fun op -> Expr.Binary (op, toExprOfType subUExprs.[0], toExprOfType subUExprs.[1])
        | UNaryOp uop ->
            match uop with
            | Discard -> Expr.Discard
            | ExtensionOp eop -> Expr.ExtensionOp (eop :?> IExtensionOp<'T>)
            |> fun op -> Expr.Nary (op, List.map toExprOfType subUExprs)

    type private ToExprOfTypeT =
        static member ToExprOfType<'T> uexpr : ExprT<'T> =
            toExprOfType uexpr

    /// converts a unified expression to an expression of the correct type
    let toExpr (UExpr (_, tn, _, _) as uexpr) =
        let gm = typeof<ToExprOfTypeT>.GetMethod ("ToExprOfType", 
                                                  BindingFlags.NonPublic ||| 
                                                  BindingFlags.Public ||| 
                                                  BindingFlags.Static)
        let m = gm.MakeGenericMethod ([| TypeName.getType tn |])
        m.Invoke(null, [| uexpr |])

    /// the op of the given unified expression
    let inline opOf uexpr =
        match uexpr with UExpr(op, typ, shp, se) -> op

    /// the type of the given unified expression
    let inline typeOf uexpr =
        match uexpr with UExpr(op, TypeName tn, shp, se) -> System.Type.GetType(tn)

    /// the type of the given unified expression
    let inline typenameOf uexpr =
        match uexpr with UExpr(op, typ, shp, se) -> typ

    /// the shape of the given unified expression
    let inline shapeOf uexpr = 
        match uexpr with UExpr(op, typ, shp, se) -> shp

    /// counts how many times subExpr occurs in unified expression uexpr
    let subExprOccurrences uexpr =
        let cnt = Dictionary<UExprT, int>()
        let rec build expr =
            if cnt.ContainsKey(expr) then
                cnt.[expr] <- cnt.[expr] + 1
            else
                cnt.[expr] <- 1

            match expr with
            | UExpr (_, _, _, srcs) ->
                for src in srcs do
                    build src
        build uexpr

        fun subExpr ->
            if cnt.ContainsKey(subExpr) then cnt.[subExpr]
            else 0

