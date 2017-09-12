namespace Models

open SymTensor

module LinearRegression =

    type Pars = {
        Weights:    ExprT 
    }

    let pars (mc: ModelBuilder<_>) nIn nOut =
        {Weights = mc.Param ("Weights", [nOut; nIn])} 
        
    let parsFromInput (mc: ModelBuilder<_>) input nOut =
        // input [smpl, inUnit]
        pars mc (Expr.shapeOf input).[1] nOut

    let pred (pars: Pars) (input: ExprT) =
        // input [smpl, inUnit]
        // pred  [smpl, outInit]
        input .* pars.Weights.T        

    let loss pars (input: ExprT) (target: ExprT) =
        let pred = pred pars input
        let two = Expr.twoOfSameType input
        (pred - target) ** two
        |> Expr.sumAxis 1
        |> Expr.mean
