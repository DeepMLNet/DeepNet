namespace SymTensor

open Basics
open ArrayNDNS
open System



module DerivCheck =

    /// evaluates the Jacobian of f at x numerically with specified finite difference step
    let inline numDerivEpsilon (epsilon: 'T) (f: ArrayNDT<'T> -> ArrayNDT<'T>) (x: ArrayNDT<'T>) =
        let y = f x
        let xElems, yElems = ArrayND.nElems x, ArrayND.nElems y
        let xShp = ArrayND.shape x

        let jac = ArrayND.zerosOfSameType [yElems; xElems] x
        let xd = x |> ArrayND.reshape [xElems] |> ArrayND.copy
        for xi = 0 to xElems - 1 do
            let xiVal = xd.[[xi]]

            // f (x+epsilon)
            xd.[[xi]] <- xiVal + epsilon
            let ydf = xd |> ArrayND.reshape xShp |> f |> ArrayND.reshape [yElems]

            // f (x-epsilon)
            xd.[[xi]] <- xiVal - epsilon
            let ydb = xd |> ArrayND.reshape xShp |> f |> ArrayND.reshape [yElems]

            // [f (x+epsilon) - f (x-epsilon)] / (2 * epsilon) 
            jac.[*, xi] <- (ydf - ydb) / (ArrayND.scalarOfSameType ydf (epsilon + epsilon))
            xd.[[xi]] <- xiVal
        jac 

    /// evaluates the Jacobian of f at x numerically
    let inline numDeriv f x = 
        numDerivEpsilon (conv<'T> 1e-5) f x

    /// Checks that symbolic and numeric derivatives of the given expression are close enough.
    /// The derivatives are evaluated at the location specified by the given VarEnv.
    let inline checkExpr (device: IDevice) (maxDeviation: 'T) (epsilon: 'T) varEnv expr =
        let rDiffs = Deriv.compute expr
        for wrt, rDiff in rDiffs.Jacobians |> Map.toSeq do
            let varEnvWithoutWrt = varEnv |> VarEnv.removeVarSpec wrt
            let exprFun = expr |> Func.make<'T> device.DefaultFactory |> addVarEnv varEnvWithoutWrt |> arg1 (Expr.makeVar wrt)
            let rDiffFun = rDiff |> Func.make<'T> device.DefaultFactory |> addVarEnv varEnvWithoutWrt |> arg1 (Expr.makeVar wrt)

            let value = VarEnv.getVarSpec wrt varEnv
            let symGradVal = rDiffFun value
            let exprGradVal = numDerivEpsilon epsilon exprFun value
            let gradDiff = abs (symGradVal - exprGradVal)

            let deviation = ArrayND.sum gradDiff |> ArrayND.value
            if deviation > maxDeviation then
                printfn "Symbolic grad of \n%s\n wrt %A is \n%s\n with value \n%A" 
                        (truncStr expr) wrt (truncStr rDiff) symGradVal
                printfn "and numeric grad has value \n%A." exprGradVal

                failwithf "Deviation of expression %s is %A which is greater than maximum deviation %A."
                    (truncStr expr) deviation maxDeviation

    /// Recursively checks that symbolic and numeric derivatives of all ops in the given expression are close enough.
    /// The derivatives are evaluated at the location specified by the given VarEnv.
    let inline checkExprTree (device: IDevice) (maxDeviation: 'T) (epsilon: 'T) (varEnv: VarEnvT) (expr: ExprT) = 
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
            checkExpr device maxDeviation epsilon varEnv expr

        checkSubExpr expr





