namespace SymTensor

open Tensor
open DeepNet.Utils

open System



module DerivCheck =

    /// evaluates the Jacobian of f at x numerically with specified finite difference step
    let inline numDerivEpsilon (epsilon: 'T) (f: Tensor<'T> -> Tensor<'T>) (x: Tensor<'T>) =
        let y = f x
        let xElems, yElems = Tensor.nElems x, Tensor.nElems y
        let xShp = Tensor.shape x

        let jac = Tensor.zeros x.Dev [yElems; xElems] 
        let xd = x |> Tensor.reshape [xElems] |> Tensor.copy
        for xi in 0L .. xElems-1L do
            let xiVal = xd.[[xi]]

            // f (x+epsilon)
            xd.[[xi]] <- xiVal + epsilon
            let ydf = xd |> Tensor.reshape xShp |> f |> Tensor.reshape [yElems]

            // f (x-epsilon)
            xd.[[xi]] <- xiVal - epsilon
            let ydb = xd |> Tensor.reshape xShp |> f |> Tensor.reshape [yElems]

            // [f (x+epsilon) - f (x-epsilon)] / (2 * epsilon) 
            jac.[*, xi] <- (ydf - ydb) / (Tensor.scalar ydf.Dev (epsilon + epsilon))
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
            if wrt.Type = typeof<'T> then
                let varEnvWithoutWrt = varEnv |> VarEnv.removeVarSpec wrt
                let exprFun = expr |> Func.make<'T> device.DefaultFactory |> addVarEnv varEnvWithoutWrt |> arg1 (Expr.makeVar wrt)
                let rDiffFun = rDiff |> Func.make<'T> device.DefaultFactory |> addVarEnv varEnvWithoutWrt |> arg1 (Expr.makeVar wrt)

                let value = VarEnv.getVarSpec wrt varEnv
                let symGradVal = rDiffFun value
                let exprGradVal = numDerivEpsilon epsilon exprFun value
                let gradDiff = abs (symGradVal - exprGradVal)

                let deviation = Tensor.sum gradDiff 
                if deviation > maxDeviation then
                    printfn "Symbolic grad of \n%s\n wrt %A is \n%s\n with value \n%A" 
                            (String.truncObj expr) wrt (String.truncObj rDiff) symGradVal
                    printfn "and numeric grad has value \n%A." exprGradVal
                    failwithf "Deviation of expression %s is %A which is greater than maximum deviation %A."
                        (String.truncObj expr) deviation maxDeviation
            else
                printfn "DerivCheck: Skipping variable %A because it does not match type %A."
                        wrt typeof<'T>

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





