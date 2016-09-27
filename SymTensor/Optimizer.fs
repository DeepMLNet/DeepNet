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
                | ElemExpr.Leaf (ElemExpr.SizeValue (ss, _)) -> ss.ContainsSymbol dimSym 
                | ElemExpr.Leaf (ElemExpr.ArgElement ((arg, sa), _)) ->
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
                    let typ = (ElemExpr.typeName elemExpr).Type
                    let sumArgPos = newArg ()
                    let sumArgIdx =
                        [for d=0 to nDims - 1 do yield ElemExpr.idx d]
                    let sumArg = ElemExpr.argElemWithType typ sumArgPos sumArgIdx

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


    and leafOpToElemOp op =
        match op with
        | ScalarConst cs        -> Some (ElemExpr.Const cs)
        | SizeValue (value, tn) -> Some (ElemExpr.SizeValue (value, tn))
        | _                     -> None

    and unaryOpToElemOp op =
        match op with
        | Negate    -> Some ElemExpr.Negate
        | Abs       -> Some ElemExpr.Abs
        | SignT     -> Some ElemExpr.SignT
        | Log       -> Some ElemExpr.Log
        | Log10     -> Some ElemExpr.Log10  
        | Exp       -> Some ElemExpr.Exp
        | Sin       -> Some ElemExpr.Sin 
        | Cos       -> Some ElemExpr.Cos 
        | Tan       -> Some ElemExpr.Tan  
        | Asin      -> Some ElemExpr.Asin
        | Acos      -> Some ElemExpr.Acos
        | Atan      -> Some ElemExpr.Atan
        | Sinh      -> Some ElemExpr.Sinh
        | Cosh      -> Some ElemExpr.Cosh
        | Tanh      -> Some ElemExpr.Tanh
        | Sqrt      -> Some ElemExpr.Sqrt 
        | Ceil      -> Some ElemExpr.Ceil
        | Floor     -> Some ElemExpr.Floor
        | Round     -> Some ElemExpr.Round
        | Truncate  -> Some ElemExpr.Truncate
        | _         -> None

    and binaryOpToElemOp op =
        match op with
        | Add       -> Some ElemExpr.Add
        | Substract -> Some ElemExpr.Substract
        | Multiply  -> Some ElemExpr.Multiply
        | Divide    -> Some ElemExpr.Divide
        | Modulo    -> Some ElemExpr.Modulo
        | Power     -> Some ElemExpr.Power
        | _         -> None

    /// combines elemwise and elements operations into one elements operation
    and combineIntoElements (expr: ExprT) : ExprT =
        let shp = Expr.shapeOf expr
        let nd = Expr.nDims expr
        let idxs = [0 .. nd-1] |> List.map ElemExpr.idx

        /// Gets the element expression for the argument, or starts a
        /// new element expression if the argument is not an element expression.
        let getArgElemExpr argExpr =
            match argExpr with
            | Nary (Elements (_, argElemExpr), argArgs) -> argElemExpr, argArgs
            | _ -> ElemExpr.argElemWithType argExpr.Type 0 idxs, [argExpr]  

        /// Joins the arguments of two element expressions and adjusts them accordingly.
        let joinArgsOfElemExprs (aElemExpr, aArgs) (bElemExpr, bArgs) =
            let rec adjust expr =
                match expr with
                | ElemExpr.Leaf (ElemExpr.ArgElement ((ElemExpr.Arg arg, idx), tn)) ->
                    ElemExpr.Leaf (ElemExpr.ArgElement ((ElemExpr.Arg (arg + List.length aArgs), idx), tn))
                | ElemExpr.Leaf _ -> expr
                | ElemExpr.Unary (op, a) -> ElemExpr.Unary (op, adjust a)
                | ElemExpr.Binary (op, a, b) -> ElemExpr.Binary (op, adjust a, adjust b)
            aElemExpr, adjust bElemExpr, aArgs @ bArgs

        match expr with
        | Leaf op ->
            match leafOpToElemOp op with
            | Some elemOp -> Expr.elements shp (ElemExpr.Leaf elemOp) []
            | None -> expr
        
        | Unary (op, aExpr) ->
            match unaryOpToElemOp op with
            | Some elemOp ->       
                let aElemExpr, aArgs = getArgElemExpr aExpr        
                let elemExpr = ElemExpr.Unary (elemOp, aElemExpr)
                Expr.elements shp elemExpr aArgs
            | None -> expr
                    
        | Binary (op, aExpr, bExpr) ->
            match binaryOpToElemOp op with
            | Some elemOp ->
                let aElemExpr, aArgs = getArgElemExpr aExpr   
                let bElemExpr, bArgs = getArgElemExpr bExpr   
                let aElemExpr, bElemExpr, abArgs = 
                    joinArgsOfElemExprs (aElemExpr, aArgs) (bElemExpr, bArgs)
                let elemExpr = ElemExpr.Binary (elemOp, aElemExpr, bElemExpr) 
                Expr.elements shp elemExpr abArgs
            | None -> expr

        // TODO: if we are an ElemExpr, merge with children
        | Nary (Elements (_, elemExpr), args) ->
            expr

        | Nary _ -> expr


    /// Optimizes an expression.
    and optimize (expr: ExprT) : ExprT =
        match optimized.TryFind expr with
        | Some opt -> opt 
        | None ->
            let opt = 
                match expr with

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

                // pull permute through broadcast
                | Unary (DoBroadcast bc, Unary (PermuteAxes perm, a)) ->
                    let bcPerm = bc |> Permutation.apply (Permutation.invert perm)
                    optimize (Unary (PermuteAxes perm, Unary (DoBroadcast bcPerm, a)))

                // pull permute, broadcast and reshape through unary elementwise ops
                | Unary (UnaryElemwiseOp as op, Unary (PermuteAxes _ as lop, a)) 
                | Unary (UnaryElemwiseOp as op, Unary (Reshape _ as lop, a)) 
                | Unary (UnaryElemwiseOp as op, Unary (DoBroadcast _ as lop, a)) ->
                    optimize (Unary (lop, Unary (op, a)))

                // pull matching permute, broadcast and reshape through binary elementwise ops
                | Binary (BinaryElemwiseOp as op, Unary (PermuteAxes _ as lopa, a),
                                                  Unary (PermuteAxes _ as lopb, b))
                | Binary (BinaryElemwiseOp as op, Unary (Reshape _ as lopa, a),
                                                  Unary (Reshape _ as lopb, b))
                | Binary (BinaryElemwiseOp as op, Unary (DoBroadcast _ as lopa, a),
                                                  Unary (DoBroadcast _ as lopb, b))
                            when lopa = lopb ->
                    optimize (Unary (lopa, Binary (op, a, b)))

                // optimize elements expressions
                | Nary (Elements (resShape, elemExpr), args) ->
                    let args = args |> List.map optimize
                    Nary (Elements (resShape, elemExpr), args)
                    |> pullSumOutOfElements
                    |> broadcastInsignificantElementsAxes

                // pass through
                | Leaf _ -> expr
                | Unary(op, a) -> Unary (op, optimize a)            
                | Binary(op, a, b) -> Binary (op, optimize a, optimize b)
                | Nary(op, es) -> Nary (op, List.map optimize es)

            // try to combine elementwise operations into an element expression
            let opt = combineIntoElements opt

            optimized.[opt] <- opt
            opt



