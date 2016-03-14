module Optimizers

open SymTensor



type GradientDescentParameters<'T> = {
    Step:           'T;
}


let gradientDescent (optPars: GradientDescentParameters<'T>) (loss: ExprT<'T>) pars  =
    let dl = Deriv.compute loss
    let dldp = dl |> Deriv.ofVar pars |> Expr.reshape (Expr.shapeOf pars)
    Expr.storeToVar pars (pars - (Expr.scalar optPars.Step) * dldp)

