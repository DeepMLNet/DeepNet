namespace SymTensor

open System.Diagnostics
open System.Reflection
open System.Collections.Generic

open Basics
open Expr


[<AutoOpen>]
module UExprTypes = 

    // int holds the position of the subuexpr that has the dynamic value
    type UExprRngSpecT = SimpleRangeSpecT<int>
    type UExprRngsSpecT = SimpleRangesSpecT<int>

    /// An operation in an unified expression.
    type IUOp =
        inherit System.IComparable

    /// ops that only occurs in unified expressions
    type UExtraOpT =
        | Subtensor of UExprRngsSpecT 
        | SetSubtensor of UExprRngsSpecT
        | Elements of ShapeSpecT * UElemExpr.UElemFuncT
        | IfThenElse
        | ExtensionExtraOp of IUOp        

    /// unified op of any arity and type
    type UOpT =
        | ULeafOp of Expr.LeafOpT
        | UUnaryOp of Expr.UnaryOpT
        | UBinaryOp of Expr.BinaryOpT
        | UNaryOp of Expr.NaryOpT
        | UExtraOp of UExtraOpT

    /// metadata for an unified expression
    type UMetadata = {
        /// the data type of the result of the generating expression
        TargetType:     TypeNameT
        /// the symbolic shape of the result of the generating expression
        TargetShape:    ShapeSpecT
        /// the numeric shape of the result of the generating expression
        TargetNShape:   NShapeSpecT
        /// the generating expression, if created from one
        Expr:           Expr.ExprT option
    }

    /// unified expression (combines all arities and types and ops cannot have expressions as parameters)    
    type [<StructuralComparison; StructuralEquality; StructuredFormatDisplay("{PrettyString}")>]
        UExprT = 
        | UExpr of UOpT * (UExprT list) * UMetadata

        member this.PrettyString =
            match this with
            | UExpr (ULeafOp uop, subs, _) -> sprintf "%A" uop 
            | UExpr (UUnaryOp uop, subs, _) -> sprintf "%A (%A)" uop subs.[0]
            | UExpr (UBinaryOp uop, subs, _) -> sprintf "%A (%A, %A)" uop subs.[0] subs.[1]
            | UExpr (UNaryOp uop, subs, _) -> sprintf "%A (%A)" uop subs
            | UExpr (UExtraOp uop, subs, _) -> sprintf "%A (%A)" uop subs

    /// An IOp that can be converted to an unified expression for compilation.
    type ICompilableOp =
        inherit IOp

        /// Should create a unified expression from the given expression.
        /// This op is always the root of the passed expression.
        /// If there is a one-to-one relationship to a unified op, call the makeOneUop function
        /// with the corresponding Uop. It will generate the apropriate unified expression.
        abstract ToUExpr: expr:ExprT -> makeOneUop:(IUOp -> UExprT) -> UExprT


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
    let rec toExprRngsSpec (srs: UExprRngsSpecT) (drs: ExprT list)  =
        if drs |> List.exists (fun dr -> Expr.typename dr <> TypeName.ofType<int>) then
            failwith "need inttype for range spec"
        match srs, drs with
        | SRSSymStartSymEnd (s, fo) :: srs, _         -> SRSSymStartSymEnd (s, fo)   :: toExprRngsSpec srs drs
        | SRSDynStartSymSize (_, f) :: srs, dr :: rdrs-> SRSDynStartSymSize (dr, f)  :: toExprRngsSpec srs rdrs
        | []                              , []        -> []
        | _                               , _         -> failwith "invalid unified subtensor spec"


module UExpr =

    type private UExprCaches = {
        UExprForExpr:       Dictionary<ExprT, UExprT>
        UExprs:             Dictionary<UExprT, UExprT>
    }

    let rec private toUExprRec (caches: UExprCaches) (expr: ExprT) =
        match caches.UExprForExpr.TryFind expr with
        | Some cached -> cached
        | None ->
            let toUExprRec = toUExprRec caches

            let metadata = {
                TargetType       = Expr.typename expr 
                TargetShape      = Expr.shapeOf expr
                TargetNShape     = Expr.shapeOf expr |> ShapeSpec.eval
                Expr             = Some expr
            }

            let leaf uop        = UExpr (ULeafOp uop, [], metadata)
            let unary uop a     = UExpr (UUnaryOp uop, [toUExprRec a], metadata)
            let binary uop a b  = UExpr (UBinaryOp uop, [toUExprRec a; toUExprRec b], metadata)
            let nary uop se     = UExpr (UNaryOp uop, se |> List.map toUExprRec, metadata)
            let extra uop se    = UExpr (UExtraOp uop, se |> List.map toUExprRec, metadata)

            let uExpr =
                match expr with
                // ops that need special handling
                | Expr.Unary (Expr.Subtensor sr, a)  ->
                    let usr, dynExprs = UExprRngsSpec.ofExprRngsSpec sr    
                    extra (Subtensor usr) (a :: dynExprs)
                | Expr.Binary (Expr.SetSubtensor sr, a, b) ->
                    let usr, dynExprs = UExprRngsSpec.ofExprRngsSpec sr   
                    extra (SetSubtensor usr) (a :: b :: dynExprs)
                | Expr.Binary (Expr.IfThenElse cond, ifTrue, ifFalse) ->
                    extra IfThenElse [ifTrue; ifFalse; cond]
                | Expr.Nary (Expr.Elements (resShape, elemExpr), se) ->
                    let nDims = ShapeSpec.nDim resShape
                    let nArgs = List.length se
                    let tn = Expr.typename expr
                    extra (Elements (resShape, UElemExpr.toUElemFunc elemExpr nDims nArgs tn)) se
                | Expr.Nary (Expr.ExtensionOp eop, se) -> 
                    match eop with
                    | :? ICompilableOp as eop ->
                        let makeOneUop uop = extra (ExtensionExtraOp uop) se
                        eop.ToUExpr expr makeOneUop
                    | _ -> nary (Expr.ExtensionOp eop) se

                // all other ops are just copied over
                | Expr.Leaf op -> leaf op
                | Expr.Unary (op, a) -> unary op a
                | Expr.Binary (op, a, b) -> binary op a b
                | Expr.Nary (op, es) -> nary op es           

            let uExpr =
                match caches.UExprs.TryFind uExpr with
                | Some cached -> cached
                | None -> uExpr

            caches.UExprForExpr.[expr] <- uExpr     
            uExpr       

    /// converts an expression to a unified expression
    let toUExpr (expr: ExprT) =
        let caches = {
            UExprForExpr    = Dictionary<ExprT, UExprT>(HashIdentity.Structural)
            UExprs          = Dictionary<UExprT, UExprT>(HashIdentity.Structural)        
        }        
        toUExprRec caches expr


    /// Returns the generating expression of a unified expression.
    /// Only works if the unified expression was created using the toUExpr function.
    let toExprOfType (UExpr (uop, subUExprs, {TargetType=tn; Expr=exprOpt})) : ExprT =
        match exprOpt with 
        | Some exprObj -> unbox exprObj 
        | None -> failwith "UExpr was not created from an Expr"

    /// Converts a unified expression to an expression of the correct type.
    /// Only works if the unified expression was created using the toUExpr function.
    let toExpr (UExpr (_, _, {TargetType=tn; Expr=exprOpt}) as uexpr) : ExprT =
        match exprOpt with 
        | Some exprObj -> unbox exprObj 
        | None -> failwith "UExpr was not created from an Expr"

    /// the op of the given unified expression
    let inline opOf (UExpr(op, se, {TargetType=tn; TargetShape=shp})) = op

    /// the type of the given unified expression
    let inline typeOf (UExpr(op, se, {TargetType=tn; TargetShape=shp})) = TypeName.getType tn

    /// the type of the given unified expression
    let inline typenameOf (UExpr(op, se, {TargetType=tn; TargetShape=shp})) = tn

    /// the shape of the given unified expression
    let inline shapeOf (UExpr(op, se, {TargetType=tn; TargetShape=shp})) = shp

    /// counts unique subexpressions in a unified expression
    let countUniqueOps uexpr  =
        let visited = HashSet<UExprT> (HashIdentity.Reference)
        let rec doCount (UExpr (_, subExprs, _) as expr) =
            if visited.Contains expr then 0
            else
                visited.Add expr |> ignore

                subExprs
                |> List.map doCount
                |> List.sum
                |> fun n -> n + 1
        doCount uexpr

    /// counts how many times subExpr occurs in unified expression uexpr
    let subExprOccurrences uexpr =
        let cnt = Dictionary<UExprT, int>(HashIdentity.Reference)
        let rec build (UExpr (_, subExprs, _) as uexpr) =
            if cnt.ContainsKey(uexpr) then
                cnt.[uexpr] <- cnt.[uexpr] + 1
            else
                cnt.[uexpr] <- 1

            for subExpr in subExprs do
                build subExpr
        build uexpr

        fun subExpr ->
            if cnt.ContainsKey(subExpr) then cnt.[subExpr]
            else 0

