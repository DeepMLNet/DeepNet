namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Backend
open Tensor.Expr


type NumDeriv =

    /// Evaluates the derivative of f at x numerically with specified finite difference step.
    /// The resulting tensor has shape fE x x1 x x2 x ... xN, where fE is the number of elements
    /// of f(x) and x1 x x2 x ... x xN is the shape of x.
    /// Note: float32 is often too imprecise to get meaningful results.
    static member calc (f: ITensor -> ITensor, x: ITensor, ?epsilon: float) =
        let epsilon = defaultArg epsilon 1e-5

        let y = f x
        assert (x.Dev = y.Dev)

        let xEpsilon = epsilon |> Tensor.scalar x.Dev |> ITensor.convertToType x.DataType
        let y2Epsilon = 2.0 * epsilon |> Tensor.scalar x.Dev |> ITensor.convertToType y.DataType

        let xd = ITensor.copy x
        let deriv = ITensor.zeros x.DataType x.Dev (y.NElems :: x.Shape)

        for xIdx in ITensor.allIdx x do
            let xRng = xIdx |> List.map Rng.Elem

            // f (x+epsilon)
            xd.[xRng] <- x.[xRng].Add xEpsilon    
            let ydf = f xd |> ITensor.flatten

            // f (x-epsilon)
            xd.[xRng] <- x.[xRng].Subtract xEpsilon    
            let ydb = f xd |> ITensor.flatten

            // [f (x+epsilon) - f (x-epsilon)] / (2 * epsilon) 
            let df = (ydf.Subtract ydb).Divide y2Epsilon
            deriv.[Rng.All :: xRng] <- df

            xd.[xRng] <- x.[xRng] 

        deriv 



type DerivCheck = 

    /// Checks that symbolic and numeric derivatives of the given expression are close enough.
    /// The derivatives are evaluated at the location specified by the given VarEnv.
    static member expr (expr: UExpr, varEnv: VarEnv, ?maxDeviation: float, ?epsilon: float, ?log: string -> unit) =
        let log = defaultArg log (printfn "%s")
        let printfn format = Printf.kprintf log format 

        let maxDeviation = defaultArg maxDeviation 1e-4

        let derivExprs = Deriv.compute expr
        for wrt in expr.Vars do

            // Calculate numeric value of derivative expression.
            let derivExpr = derivExprs.Wrt wrt
            let derivExprVal = derivExpr |> UExpr.eval varEnv

            // Calculate numeric (finite difference) derivative.
            let exprFn (wrtVal: ITensor) =
                expr |> UExpr.eval (varEnv |> VarEnv.addBaseVar wrt wrtVal)
            let wrtVal = varEnv.[wrt.Name]
            let derivNumVal = NumDeriv.calc (exprFn, wrtVal, ?epsilon=epsilon)

            // Check.
            if derivExprVal.Dev <> derivNumVal.Dev then
                failwithf "Symbolic derivative has device %A but numeric derivative has device %A."
                          derivExprVal.Dev derivNumVal.Dev
            if derivExprVal.Shape <> derivNumVal.Shape then
                failwithf "Symbolic derivative has shape %A but numeric derivative has shape %A."
                          derivExprVal.Dev derivNumVal.Dev
            if derivExprVal.DataType <> derivNumVal.DataType then
                failwithf "Symbolic derivative has data type %A but numeric derivative has shape %A."
                          derivExprVal.DataType derivNumVal.DataType

            // Compare.
            let funElems = derivExprVal.Shape.[0]
            let derivDiff = (derivExprVal.Subtract derivNumVal).Abs()
            let derivDiffSum = derivDiff.SumTensor() |> ITensor.convert<float> |> Tensor.value
            let derivDiffAvg = derivDiffSum / (float funElems)

            if derivDiffAvg > maxDeviation then
                printfn "Derivative check failed for expression:\n%A\n" expr
                printfn "wrt %A:\n%A\n" wrt derivExpr
                printfn "Symbolic value:\n%A\n" derivExprVal
                printfn "Numeric value:\n%A\n" derivNumVal
                failwithf "Average difference between symbolic and numeric derivative of %A wrt. %A is %f > %f."
                          expr wrt derivDiffAvg maxDeviation

