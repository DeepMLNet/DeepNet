namespace SymTensor

open Basics
open Expr


module Optimizer =
   
    /// Cache of optimized expressions.
    let private optimized = Dictionary<ExprT, ExprT> (HashIdentity.Reference)

    /// Returns a list containing one element each axis of the expression.
    /// The element is true if the axis is broadcasted.
    let axesBroadcasted expr =
        match expr with
        | Unary (DoBroadcast bc, a) ->
            List.zip bc (Expr.shapeOf a)
            |> List.map (fun (bcDim, aDim) -> 
                aDim = SizeSpec.broadcastable && bcDim <> aDim)
        | _ ->
            Expr.shapeOf expr
            |> List.map (fun _ -> false)


    /// Returns a list containing one element for each axis in the result of the elements expression.
    /// The element is true if the result can have different values over the corresponding axis.
    /// I.e. an axis is significant if values change depending on it.
    let elementsAxesSignificancy elements =
        match elements with 
        | Nary (Elements (resShape, elemExpr), args) ->

            let argsBroadcasted =
                List.indexed args
                |> List.map (fun (argPos, arg) -> ElemExpr.Arg argPos, axesBroadcasted arg)
                |> Map.ofList

            let rec sigInDim dim elemExpr =
                let dimSym = ElemExpr.idxSymbol dim
                match elemExpr with
                | ElemExpr.Leaf (ElemExpr.SizeValue ss) -> ss.ContainsSymbol dimSym 
                | ElemExpr.Leaf (ElemExpr.ArgElement (arg, sa)) ->
                    List.zip sa argsBroadcasted.[arg]
                    |> List.exists (fun (dimSs, dimBc) ->
                        dimSs.ContainsSymbol dimSym && not dimBc)
                | ElemExpr.Leaf _ -> false

                | ElemExpr.Unary (ElemExpr.Sum (_, first, last), a) ->
                    first.ContainsSymbol dimSym 
                    || last.ContainsSymbol dimSym
                    || sigInDim dim a
                | ElemExpr.Unary (ElemExpr.KroneckerRng (ss, first, last), a) ->
                    ss.ContainsSymbol dimSym
                    || first.ContainsSymbol dimSym
                    || last.ContainsSymbol dimSym
                    || sigInDim dim a
                | ElemExpr.Unary (_, a) ->
                    sigInDim dim a

                | ElemExpr.Binary (ElemExpr.IfThenElse (left, right), a, b) ->
                    left.ContainsSymbol dimSym
                    || right.ContainsSymbol dimSym
                    || sigInDim dim a || sigInDim dim b
                | ElemExpr.Binary (_, a, b) ->
                    sigInDim dim a || sigInDim dim b

            let nDims = List.length resShape
            [0 .. nDims-1]
            |> Seq.map (fun dim -> sigInDim dim elemExpr)

        | _ -> failwith "not an elements expression"



    /// pulls summations out of elements expressions
    let rec pullSumOutOfElements elements =
        match elements with
        | Nary (Elements (resShape, elemExpr), args) ->

            let nDims = ShapeSpec.nDim resShape
            let mutable nArgs = args.Length 
            let newArg () =
                let res = nArgs
                nArgs <- nArgs + 1
                res

            let rec splitSum elemExpr =
                match elemExpr with
                | ElemExpr.Unary (ElemExpr.Sum (sym, first, last), summand) ->     
                    //printfn "Pulling out summand:\n%A" summand
                                 
                    // replace sum by argument access
                    let sumArgPos = newArg ()
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

                    // build sumand elements expression and sum over last dimensions
                    let summandExpr = Expr.elements sumandShape summandSubst args
                    let summedExpr = summandExpr |> Expr.sumAxis nDims

                    sumArg, [optimize summedExpr]

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
            Expr.elements resShape elemExprWithoutSum (args @ sumArgs)

        | _ -> failwith "not an elements expression"

    /// replaces result dimensions that compute identical values for all elements 
    /// by broadcasting
    and broadcastInsignificantElementsAxes elements =
        match elements with
        | Nary (Elements (resShape, elemExpr), args) ->
            // determine significancy of all axes
            let axSigs = elementsAxesSignificancy elements
            let insigAx = 
                Seq.indexed axSigs 
                |> Seq.tryFind (fun (ax, axSig) ->
                    not axSig && resShape.[ax] .<> SizeSpec.broadcastable)

            match insigAx with
            | Some (insigAx, _) ->
                //printfn "removing insignificant axis %d with shape %A of expr:\n%A"
                //    insigAx resShape.[insigAx] elemExpr

                // replace insignificant axis by axis with one broadcastable element
                let sigResShape = resShape |> ShapeSpec.set insigAx SizeSpec.broadcastable
                let sigElements = Expr.elements sigResShape elemExpr args

                // broadcast result to original shape
                let bcElements = sigElements |> Expr.broadcast resShape
                optimize bcElements
            | None -> elements

        | _ -> failwith "not an elements expression"



    /// Optimizes an expression.
    and optimize (expr: ExprT) : ExprT =
        match optimized.TryFind expr with
        | Some opt -> opt 
        | None ->
            let opt = 
                match expr with
                | Leaf _ -> expr

                // remove unnecessary axes permutations
                | Unary (PermuteAxes perm, a) when Permutation.isIdentity perm ->
                    optimize a

                // remove unnecessary reshapes
                | Unary (Reshape ss, a) when ShapeSpec.equalWithBroadcastability ss (shapeOf a) ->
                    optimize a            

                // remove unnecessary broadcasts
                | Unary (DoBroadcast ss, a) when ShapeSpec.equalWithBroadcastability ss (shapeOf a) ->
                    optimize a

                // combine subsequent axes permutations
                | Unary (PermuteAxes perm1, Unary (PermuteAxes perm2, a)) ->
                    let perm = Permutation.chain perm1 perm2
                    optimize (Unary (PermuteAxes perm, a))

                // combine subsequent reshapes
                | Unary (Reshape ss, Unary (Reshape _, a)) ->
                    optimize (Unary (Reshape ss, a))

                // combine subsequent broadcasts
                | Unary (DoBroadcast bc, Unary (DoBroadcast _, a)) ->
                    optimize (Unary (DoBroadcast bc, a))

                | Unary(op, a) -> Unary (op, optimize a)            
                | Binary(op, a, b) -> Binary (op, optimize a, optimize b)

                // optimize elements expressions
                | Nary (Elements (resShape, elemExpr), args) ->
                    let args = args |> List.map optimize
                    Nary (Elements (resShape, elemExpr), args)
                    |> pullSumOutOfElements
                    |> broadcastInsignificantElementsAxes

                | Nary(op, es) -> Nary (op, List.map optimize es)

            optimized.[opt] <- opt
            opt




