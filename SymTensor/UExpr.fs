namespace SymTensor

open System.Reflection
open System.Collections.Generic
open Expr
open UExprTypes


module UVarSpec =

    /// create variable specifation by name and shape
    let inline ofNameShapeAndTypeName name shape typeName : UVarSpecT =
        {Name=name; Shape=shape; TypeName=typeName;}

    let ofVarSpec (vs: #IVarSpec) =
        {Name=vs.Name; Shape=vs.Shape; TypeName=vs.TypeName}

    let ofExpr expr =
        expr |> Expr.extractVar |> ofVarSpec

    let toVarSpec (vs: UVarSpecT) : VarSpecT<'T> =
        {Name=vs.Name; Shape=vs.Shape;}

    let name (vs: UVarSpecT) =
        vs.Name

    let shape (vs: UVarSpecT) =
        vs.Shape

    let nDims vs =
        shape vs |> List.length

    let typ (vs: UVarSpecT) = 
        vs.TypeName |> TypeName.getType 

    let substSymSizes symSizes (vs: UVarSpecT) = 
        {vs with Shape=SymSizeEnv.substShape symSizes vs.Shape} 

    let tryFindByName (vs: UVarSpecT) map =
        map |> Map.tryPick 
            (fun cvs value -> 
                if name cvs = name vs then Some value
                else None)

    let findByName vs map =
        match tryFindByName vs map with
        | Some value -> value
        | None -> raise (KeyNotFoundException())


module UExprRngsSpec =

    // split into two ops:
    // one that does nothing, just changes the static layout
    // and another that does the copying if necessary
    // op1 : StaticSubtensor
    // op2 : DynamicSubtensor
    // but how does this work with SetSubtensor?

    /// converts a ExprRngsSpecT to a UExprRngSpecT
    let ofExprRngsSpec (sr: ExprRngsSpecT) =
        ([], sr)
        ||> List.mapFold (fun dynExprs rng ->
            let idx = List.length dynExprs 
            match rng with
            | SRSSymStartSymEnd  (s, fo)     -> SRSSymStartSymEnd (s, fo),       dynExprs
            | SRSDynStartSymSize (s, size)   -> SRSDynStartSymSize (idx, size),  dynExprs @ [s])

    /// converts a UExprRngSpecT to a ExprRngsSpecT
    let rec toExprRngsSpec (srs: UExprRngsSpecT) (drs: ExprT<int> list)  =
        match srs, drs with
        | SRSSymStartSymEnd (s, fo) :: srs, _         -> SRSSymStartSymEnd (s, fo)   :: toExprRngsSpec srs drs
        | SRSDynStartSymSize (_, f) :: srs, dr :: rdrs-> SRSDynStartSymSize (dr, f)  :: toExprRngsSpec srs rdrs
        | []                              , []        -> []
        | _                               , _         -> failwith "invalid unified subtensor spec"


module UExpr =

    /// converts an expression to a unified expression
    let rec toUExpr (expr: ExprT<'T>) =
        let metadata = {
            TargetType       = TypeName typeof<'T>.AssemblyQualifiedName 
            TargetShape      = Expr.shapeOf expr
            TargetNShape     = Expr.shapeOf expr |> ShapeSpec.eval
            Expr             = Some (expr :> System.IComparable)
        }

        let leaf uop        = UExpr (ULeafOp uop, [], metadata)
        let unary uop a     = UExpr (UUnaryOp uop, [toUExpr a], metadata)
        let binary uop a b  = UExpr (UBinaryOp uop, [toUExpr a; toUExpr b], metadata)
        let nary uop se     = UExpr (UNaryOp uop, se |> List.map toUExpr, metadata)

        match expr with
        | Leaf (Expr.Identity ss)       -> leaf (Identity ss)
        | Leaf (Expr.Zeros ss)          -> leaf (Zeros ss)
        | Leaf (Expr.ScalarConst v)     -> leaf (ScalarConst (box v :?> System.IComparable))
        | Leaf (Expr.SizeValue sv)      -> leaf (SizeValue sv)
        | Leaf (Expr.Var vs)            -> leaf (Var (UVarSpec.ofVarSpec vs))

        | Unary (Expr.Negate, a)        -> unary Negate a
        | Unary (Expr.Abs, a)           -> unary Abs a
        | Unary (Expr.SignT, a)         -> unary SignT a
        | Unary (Expr.Log, a)           -> unary Log a
        | Unary (Expr.Log10, a)         -> unary Log10 a
        | Unary (Expr.Exp, a)           -> unary Exp a
        | Unary (Expr.Sin, a)           -> unary Sin a
        | Unary (Expr.Cos, a)           -> unary Cos a
        | Unary (Expr.Tan, a)           -> unary Tan a
        | Unary (Expr.Asin, a)          -> unary Asin a
        | Unary (Expr.Acos, a)          -> unary Acos a
        | Unary (Expr.Atan, a)          -> unary Atan a
        | Unary (Expr.Sinh, a)          -> unary Sinh a
        | Unary (Expr.Cosh, a)          -> unary Cosh a
        | Unary (Expr.Tanh, a)          -> unary Tanh a
        | Unary (Expr.Sqrt, a)          -> unary Sqrt a
        | Unary (Expr.Ceil, a)          -> unary Ceil a
        | Unary (Expr.Floor, a)         -> unary Floor a
        | Unary (Expr.Round, a)         -> unary Round a
        | Unary (Expr.Truncate, a)      -> unary Truncate a
        | Unary (Expr.Diag (ax1, ax2), a) -> unary (Diag (ax1, ax2)) a
        | Unary (Expr.DiagMat (ax1, ax2), a)  -> unary (DiagMat (ax1, ax2)) a
        | Unary (Expr.Invert, a)        -> unary Invert a
        | Unary (Expr.Sum, a)           -> unary Sum a
        | Unary (Expr.SumAxis ax, a)    -> unary (SumAxis ax) a
        | Unary (Expr.Reshape ss, a)    -> unary (Reshape ss) a
        | Unary (Expr.DoBroadcast ss, a)-> unary (DoBroadcast ss) a
        | Unary (Expr.SwapDim (ax1, ax2), a) -> unary (SwapDim (ax1, ax2)) a
        | Unary (Expr.Subtensor sr, a)  ->
            let usr, dynExprs = UExprRngsSpec.ofExprRngsSpec sr    
            let dynUExprs = dynExprs |> List.map toUExprForInt               
            UExpr(UNaryOp (Subtensor usr), toUExpr a :: dynUExprs, metadata)
        | Unary (Expr.StoreToVar vs, a) -> unary (StoreToVar (UVarSpec.ofVarSpec vs)) a
        | Unary (Expr.Annotated ano, a) -> unary (Annotated ano) a

        | Binary (Expr.Add, a, b)       -> binary Add a b
        | Binary (Expr.Substract, a, b) -> binary Substract a b
        | Binary (Expr.Multiply, a, b)  -> binary Multiply a b                     
        | Binary (Expr.Divide, a, b)    -> binary Divide a b             
        | Binary (Expr.Modulo, a, b)    -> binary Modulo a b          
        | Binary (Expr.Power, a, b)     -> binary Power a b               
        | Binary (Expr.Dot, a, b)       -> binary Dot a b                   
        | Binary (Expr.TensorProduct, a, b) -> binary TensorProduct a b         
        | Binary (Expr.SetSubtensor sr, a, b) ->
            let usr, dynExprs = UExprRngsSpec.ofExprRngsSpec sr    
            let dynUExprs = dynExprs |> List.map toUExprForInt 
            UExpr(UNaryOp (SetSubtensor usr), toUExpr a :: toUExpr b :: dynUExprs, 
                  metadata)

        | Nary (Expr.Discard, se)       -> nary Discard se
        | Nary (Expr.Elements (resShape, elemExpr), se) ->
            let nDims = ShapeSpec.nDim resShape
            let nArgs = List.length se
            nary (Elements (resShape, UElemExpr.toUElemFunc elemExpr nDims nArgs)) se
        | Nary (Expr.ExtensionOp eop, se) -> 
            let makeOneUop uop = nary (ExtensionOp uop) se
            eop.ToUExpr expr makeOneUop
            

    and private toUExprForInt (expr: ExprT<int>) =
        toUExpr expr

    /// Returns the generating expression of a unified expression.
    /// Only works if the unified expression was created using the toUExpr function.
    let toExprOfType (UExpr (uop, subUExprs, {TargetType=tn; Expr=exprOpt})) : ExprT<'T> =
        if TypeName.ofType<'T> <> tn then
            failwith "UExpr type does not match"

        match exprOpt with 
        | Some exprObj -> unbox exprObj 
        | None -> failwith "UExpr was not created from an Expr"

    type private ToExprOfTypeT =
        static member ToExprOfType<'T> uexpr : ExprT<'T> =
            toExprOfType uexpr

    /// Converts a unified expression to an expression of the correct type.
    /// Only works if the unified expression was created using the toUExpr function.
    let toExpr (UExpr (_, _, {TargetType=tn}) as uexpr) =
        let gm = typeof<ToExprOfTypeT>.GetMethod ("ToExprOfType", 
                                                  BindingFlags.NonPublic ||| 
                                                  BindingFlags.Public ||| 
                                                  BindingFlags.Static)
        let m = gm.MakeGenericMethod ([| TypeName.getType tn |])
        m.Invoke(null, [| uexpr |])

    /// the op of the given unified expression
    let inline opOf (UExpr(op, se, {TargetType=tn; TargetShape=shp})) = op

    /// the type of the given unified expression
    let inline typeOf (UExpr(op, se, {TargetType=tn; TargetShape=shp})) = TypeName.getType tn

    /// the type of the given unified expression
    let inline typenameOf (UExpr(op, se, {TargetType=tn; TargetShape=shp})) = tn

    /// the shape of the given unified expression
    let inline shapeOf (UExpr(op, se, {TargetType=tn; TargetShape=shp})) = shp

    /// counts how many times subExpr occurs in unified expression uexpr
    let subExprOccurrences uexpr =
        let cnt = Dictionary<UExprT, int>()
        let rec build expr =
            if cnt.ContainsKey(expr) then
                cnt.[expr] <- cnt.[expr] + 1
            else
                cnt.[expr] <- 1

            match expr with
            | UExpr (_, srcs, _) ->
                for src in srcs do
                    build src
        build uexpr

        fun subExpr ->
            if cnt.ContainsKey(subExpr) then cnt.[subExpr]
            else 0

