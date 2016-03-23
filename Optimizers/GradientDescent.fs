namespace Optimizers

open SymTensor

module GradientDescent =

    type Pars<'T> = {
        Step:           'T;
    }

    let minimize (optPars: Pars<'T>) (loss: ExprT<'T>) pars  =
        let dl = Deriv.compute loss
        let dldp = dl |> Deriv.ofVar pars |> Expr.reshape (Expr.shapeOf pars)
        Expr.storeToVar pars (pars - (Expr.scalar optPars.Step) * dldp)

    

