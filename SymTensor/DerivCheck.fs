namespace SymTensor

open Basics
open ArrayNDNS
open System



module DerivCheck =

    /// evaluates the Jacobian of f at x numerically with specified finite difference step
    let inline numDerivEpsilon (epsilon: ^T) (f: ArrayNDT<'T> -> ArrayNDT<'T>) (x: ArrayNDT<'T>) =
        let y = f x
        let xShp, yShp = ArrayND.shape x, ArrayND.shape y
        let xElems, yElems = ArrayND.nElems x, ArrayND.nElems y
        let xf, yf = x |> ArrayND.reshape [xElems], y |> ArrayND.reshape [yElems]

        let j = ArrayNDHost.zeros [yElems; xElems]
        for xi = 0 to xElems - 1 do
            let xdf = ArrayND.copy xf
            xdf |> ArrayND.set [xi] ((xf |> ArrayND.get [xi]) + epsilon)
            let ydf = xdf |> ArrayND.reshape xShp |> f |> ArrayND.reshape [yElems]
            let d : ArrayNDT<'T> = (ydf - yf) / (ArrayND.scalarOfType epsilon ydf)
            ArrayND.copyTo d (j |> ArrayND.view [RngAll; RngElem xi])
        j

    /// evaluates the Jacobian of f at x numerically
    let inline numDeriv f x = 
        numDerivEpsilon (conv<'T> 1e-5) f x

    /// Checks that symbolic and numeric derivatives of the given expression are close enough.
    /// The derivatives are evaluated at the location specified by the given VarEnv.
    let inline checkExpr (maxDeviation: 'T) (epsilon: 'T) varEnv expr =
        let rDiffs = Deriv.compute expr
        for wrt, rDiff in rDiffs |> Map.toSeq do
            let varEnvWithoutWrt = varEnv |> VarEnv.removeVarSpec wrt
            let exprFun = expr |> Func.make<'T> DevHost.DefaultFactory |> addVarEnv varEnvWithoutWrt |> arg1 (Expr.makeVar wrt)
            let rDiffFun = rDiff |> Func.make<'T> DevHost.DefaultFactory |> addVarEnv varEnvWithoutWrt |> arg1 (Expr.makeVar wrt)

            let value = VarEnv.getVarSpec wrt varEnv
            let symGradVal = rDiffFun value
            let exprGradVal = numDerivEpsilon epsilon exprFun value
            let gradDiff = abs (symGradVal - exprGradVal)

            let deviation = ArrayND.sum gradDiff |> ArrayND.value
            if deviation > maxDeviation then
                printfn "Symbolic grad of \n%A\n wrt %A is \n%A\n with value \n%A" expr wrt rDiff symGradVal
                printfn "and numeric grad has value \n%A." exprGradVal

                failwithf "Deviation of expression %A is %A which is greater than maximum deviation %A."
                    expr deviation maxDeviation

    /// Recursively checks that symbolic and numeric derivatives of all ops in the given expression are close enough.
    /// The derivatives are evaluated at the location specified by the given VarEnv.
    let inline checkExprTree (maxDeviation: 'T) (epsilon: 'T) (varEnv: VarEnvT) (expr: ExprT) = 
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
            checkExpr maxDeviation epsilon varEnv expr

        checkSubExpr expr





