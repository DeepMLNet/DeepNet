namespace Models

open SymTensor

module LinearRegression =

    type Pars = {
        Weights:    ExprT ref
    }

    let pars (mc: ModelBuilder<_>) nIn nOut =
        {Weights = mc.Param ("Weights", [nOut; nIn])} 
        
    let parsFromInput (mc: ModelBuilder<_>) input nOut =
        pars mc (Expr.shapeOf input).[0] nOut

    let pred (pars: Pars) (input: ExprT) =
        !pars.Weights .* input

    let loss pars (input: ExprT) (target: ExprT) =
        let pred = pred pars input
        let two = Expr.twoOfSameType input
        let diff = (pred - target) ** two
        Expr.sum diff
