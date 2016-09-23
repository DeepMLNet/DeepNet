namespace SymTensor

open Basics
open Expr


module Optimizer =
  
    /// optimizes an elements expression
    let rec optimizeElements elements =
        match elements with
        | Nary (Elements (resShape, elemExpr), es) ->
            let nDims = ShapeSpec.nDim resShape
            let mutable nArgs = es.Length 
            let nextArg () =
                let res = nArgs
                nArgs <- nArgs + 1
                res

            let rec splitSum elemExpr =
                match elemExpr with
                | ElemExpr.Unary (ElemExpr.Sum (sym, first, last), summand) ->                  
                    // replace sum by argument access
                    let sumArgPos = nextArg ()
                    let sumArgIdx =
                        [for d=0 to nDims - 1 do yield ElemExpr.idx d]
                    let sumArg = ElemExpr.argElem sumArgPos sumArgIdx

                    // add summation dimension to the right
                    let sumElems = last - first - 1
                    let sumandShape = resShape @ [sumElems]
                    let sumandIdx = first + ElemExpr.idx nDims

                    // substitute summation symbol with last dimension index
                    let subst = Map [sym, sumandIdx]
                    let summandSubst = summand |> ElemExpr.substSymSizes subst

                    // build sumand elements expression and optimize
                    let summandExpr = 
                        Expr.elements sumandShape summandSubst es
                        |> optimizeElements

                    // sum over last dimension
                    let summedExpr = summandExpr |> Expr.sumAxis nDims

                    sumArg, [summedExpr]

                | ElemExpr.Leaf (op) -> 
                    ElemExpr.Leaf (op), []
                | ElemExpr.Unary (op, a) ->
                    let aSplit, aArgs = splitSum a
                    ElemExpr.Unary (op, aSplit), aArgs
                | ElemExpr.Binary (op, a, b) ->
                    let aSplit, aArgs = splitSum a
                    let bSplit, bArgs = splitSum b
                    ElemExpr.Binary (op, aSplit, bSplit), aArgs @ bArgs

            let elemExprWithoutSum, sumArgs = splitSum elemExpr
            Expr.elements resShape elemExprWithoutSum (es @ sumArgs)

        | _ -> failwith "not an elements expression"

    /// Cache of optimized expressions.
    let private optimized = Dictionary<System.IComparable, obj> (HashIdentity.Reference)

    /// Optimizes an expression.
    let rec optimize (expr: ExprT<'T>) : ExprT<'T> =
        match optimized.TryFind expr with
        | Some opt -> opt :?> ExprT<'T>
        | None ->
            let opt = 
                match expr with
                | Leaf _ -> expr

                // combine subsequent reshapes
                | Unary (Reshape ss, Unary (Reshape _, a)) ->
                    optimize (Unary(Reshape ss, a))

                // remove unnecessary reshapes
                | Unary (Reshape ss, a) when ShapeSpec.equalWithBroadcastability ss (shapeOf a) ->
                    optimize a            

                | Unary(op, a) -> Unary(op, optimize a)            
                | Binary(op, a, b) -> Binary(op, optimize a, optimize b)

                | Nary (Elements (resShape, elemExpr), es) ->
                    let es = es |> List.map optimize
                    optimizeElements (Nary (Elements (resShape, elemExpr), es))

                | Nary(op, es) -> Nary(op, List.map optimize es)

            optimized.[opt] <- opt
            opt




