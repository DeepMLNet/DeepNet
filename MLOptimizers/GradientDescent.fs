namespace Optimizers

open Basics
open ArrayNDNS
open SymTensor

    
module GradientDescent = 

    type Cfg<'T> = {
        Step:           'T
    }

    type CfgExpr<'T> = {
        Step:           ExprT<'T>
    }


open GradientDescent

type GradientDescent<'T when 'T: equality> (loss:   ExprT<'T>,
                                            pars:   ExprT<'T>,    
                                            dev:    IDevice) =

    let cfg = {
        Step        = Expr.var "GradientDescent.Cfg.Step" []
    }

    let rp = VarRecord<Cfg<'T>, CfgExpr<'T>> (cfg, dev)

    member this.Minimize =
        let grad = Deriv.compute loss |> Deriv.ofVar pars |> Expr.reshape (Expr.shapeOf pars)
        Expr.storeToVar pars (pars - cfg.Step * grad)

    member this.Use f =
        rp.Use f

    member this.PublishLoc mb =
        rp.PublishLoc mb




