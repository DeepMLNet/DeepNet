namespace Models

open SymTensor

module LinearRegression =

    type Pars<'T> = {
        Weights:    ExprT<'T>;
    }

    let pars (mc: ModelBuilder<_>) nIn nOut =
        {Weights = mc.Param "Weights"     [nOut; nIn]} 
        
    let parsFromInput (mc: ModelBuilder<_>) input nOut =
        pars mc (Expr.shapeOf input).[0] nOut

    let pred (pars: Pars<'T>) (input: ExprT<'T>) =
        pars.Weights .* input

    let loss pars (input: ExprT<'T>) (target: ExprT<'T>) =
        let pred = pred pars input
        let diff = (pred - target) ** Expr.two<'T>()
        Expr.sum diff
