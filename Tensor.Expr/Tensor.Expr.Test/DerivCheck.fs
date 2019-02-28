namespace Tensor.Expr

open Tensor
open Tensor.Backend
open Tensor.Expr
open DeepNet.Utils

//open System



type NumDeriv =

    /// Evaluates the derivative of f at x numerically with specified finite difference step.
    /// The resulting tensor has shape fE x x1 x x2 x ... xN, where fE is the number of elements
    /// of f(x) and x1 x x2 x ... x xN is the shape of x.
    static member calc (f: Tensor<'T> -> Tensor<'T>, x: Tensor<'T>, ?epsilon: float) =
        let epsilon = defaultArg epsilon 1e-5
        let epsilon = Tensor.scalar x.Dev (conv<'T> epsilon)

        let y = f x
        let xd = Tensor.copy x

        let deriv = Tensor.zeros x.Dev (y.NElems :: x.Shape)
        for xIdx in Tensor.allIdx x do
            let xRng = xIdx |> List.map Rng.Elem

            // f (x+epsilon)
            xd.[xRng] <- x.[xRng] + epsilon    
            let ydf = f xd |> Tensor.flatten

            // f (x-epsilon)
            xd.[xRng] <- x.[xRng] - epsilon    
            let ydb = f xd |> Tensor.flatten

            // [f (x+epsilon) - f (x-epsilon)] / (2 * epsilon) 
            let df = (ydf - ydb) / (epsilon + epsilon)
            deriv.[Rng.All :: xRng] <- df

            xd.[xRng] <- x.[xRng] 

        deriv 



type DerivCheck = 

    /// Checks that symbolic and numeric derivatives of the given expression are close enough.
    /// The derivatives are evaluated at the location specified by the given VarEnv.
    static member expr (expr: Expr, varEnv: VarEnv, ?maxDeviation: float, ?epsilon: float) =
        let maxDeviation = defaultArg maxDeviation 1e-4

        let derivExprs = Deriv.compute expr
        for wrt in expr.Vars do

            // Calculate numeric value of derivative expression.
            let derivExpr = derivExprs.[wrt]
            let derivExprVal = derivExpr |> Expr.eval varEnv

            // Calculate numeric (finite difference) derivative.
            let exprFn (wrtVal: ITensor) =
                expr |> Expr.eval (varEnv |> VarEnv.add wrt wrtVal)
            let wrtVal = varEnv |> VarEnv.get wrt
            let derivNumVal = NumDeriv.calc (exprFn, wrtVal, ?epsilon=epsilon)



            let varEnvWithoutWrt = varEnv |> VarEnv.removeVarSpec wrt
            let exprFun = expr |> Func.make<'T> device.DefaultFactory |> addVarEnv varEnvWithoutWrt |> arg1 (Expr.makeVar wrt)
            let rDiffFun = derivExpr |> Func.make<'T> device.DefaultFactory |> addVarEnv varEnvWithoutWrt |> arg1 (Expr.makeVar wrt)

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

            //else
            //    printfn "DerivCheck: Skipping variable %A because it does not match type %A."
            //            wrt typeof<'T>
