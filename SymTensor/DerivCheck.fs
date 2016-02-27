namespace SymTensor

open ArrayNDNS
open System


module DerivCheck =


    /// evaluates the Jacobian of f at x numerically with specified finite difference step
    let inline numGradEpsilon (epsilon: ^T) (f: ArrayNDT<'T> -> ArrayNDT<'T>) (x: ArrayNDT<'T>) =
        let y = f x
        let xShp, yShp = ArrayND.shape x, ArrayND.shape y
        let xElems, yElems = ArrayND.nElems x, ArrayND.nElems y
        let xf, yf = x |> ArrayND.reshape [xElems], y |> ArrayND.reshape [yElems]

        let j = ArrayNDHost.zeros [yElems; xElems]
        for xi = 0 to xElems - 1 do
            let xdf = ArrayND.copy xf
            xdf |> ArrayND.set [xi] ((xf |> ArrayND.get [xi]) + epsilon)
            let ydf = xdf |> ArrayND.reshape xShp |> f |> ArrayND.reshape [yElems]
            let d = (ydf - yf) / epsilon       
            j |> ArrayND.view [RngAll; RngElem xi] |> ArrayND.copyTo d
        j

    /// evaluates the Jacobian of f at x numerically
    let inline numGrad (f: ArrayNDT<'T> -> ArrayNDT<'T>) (x: ArrayNDT<'T>) = 
        let epsilon = Convert.ChangeType(1e-5, typeof<'T>) :?> 'T
        numGradEpsilon epsilon f x

    //let exprGradDiff evalEnv wrt expr =
    //    let g = ExprForwardDiff.grad wrt expr
    //    let exprFun = (expr |> OpEval.toFun |> OpEval.addArg wrt) |> OpEval.usingEvalEnv evalEnv
    //    let gradFun = (g |> OpEval.toFun |> OpEval.addArg wrt) |> OpEval.usingEvalEnv evalEnv
    //
    //    let value = evalEnv.VarEnv.[Op.extractVar wrt]
    //    let symGradVal = gradFun value
    //    let exprGradVal = numGrad exprFun value
    //    let gradDiff = abs (symGradVal - exprGradVal)
    //    sum gradDiff |> NDArray.value


    let inline reverseDiffDeviations evalEnv expr =
        let mutable devs = Map.empty
        let rDiffs = Deriv.compute expr
        for wrt, rDiff in rDiffs |> Map.toSeq do
            let exprFun = (expr |> Eval.toFun |> Eval.addArg (Expr.makeVar wrt)) |> Eval.usingEvalEnv evalEnv
            let rDiffFun = (rDiff |> Eval.toFun |> Eval.addArg (Expr.makeVar wrt)) |> Eval.usingEvalEnv evalEnv

            let value = EvalEnv.getVarSpecT wrt evalEnv
            let symGradVal = rDiffFun value
            let exprGradVal = numGrad exprFun value
            let gradDiff = abs (symGradVal - exprGradVal)

            devs <- devs |> Map.add (VarSpec.name wrt) (ArrayND.sum gradDiff |> ArrayND.value)

            //printfn "Symbolic grad of \n%A\n wrt %A is \n%A\n with value \n%A" expr wrt rDiff symGradVal
            //printfn "and numeric grad has value \n%A." exprGradVal

        devs

    let inline reverseDiffDeviationsOkay evalEnv (expr: ExprT<'T>) =
        let maxDeviation = Convert.ChangeType(1e-4, typeof<'T>) :?> 'T
        let devs = reverseDiffDeviations evalEnv expr
        devs |> Map.iter
            (fun name dev -> if dev > maxDeviation then printfn "deviation wrt %A = %A" name dev)
        devs |> Map.forall (fun _ dev -> dev < maxDeviation) 


    let inline checkReverseDiff (evalEnv: EvalEnvT) (expr: ExprT<'T>) = 
        let evalEnv = evalEnv |> EvalEnv.enhance VarEnv.empty (Seq.singleton expr)

        let rec checkSubExpr expr = 
            match expr with
            | Expr.Leaf(_) -> ()
            | Expr.Unary(_, a) -> 
                checkSubExpr a
            | Expr.Binary(_, a, b) -> 
                checkSubExpr a
                checkSubExpr b
            | Expr.Nary(_, es) ->
                es |> List.iter checkSubExpr

            if not (reverseDiffDeviationsOkay evalEnv expr) then
                failwithf "deviation between numeric and symbolic derivative too large in op %A" (UExpr.extractOp expr)

        checkSubExpr expr





