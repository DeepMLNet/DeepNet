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


[<AutoOpen>]
module GradientDescentTypes = 
    open GradientDescent

    type GradientDescent<'T when 'T: equality> (dev: IDevice) =
        let rp = RecordParams<Cfg<'T>, CfgExpr<'T>> dev
        let optPars = rp.Expr

        member this.Minimize (loss: ExprT<'T>) pars  =
            let dl = Deriv.compute loss
            let dldp = dl |> Deriv.ofVar pars |> Expr.reshape (Expr.shapeOf pars)
            Expr.storeToVar pars (pars - optPars.Step * dldp)

        member this.Cfg f =
            rp.Use f

        member this.PublishCfgLoc mb =
            rp.PublishLoc mb




