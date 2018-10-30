namespace Optimizers

open DeepNet.Utils
open Tensor
open SymTensor

    
module GradientDescent = 

    type Cfg<'T> = {
        Step:           'T
    }

    type CfgExpr = {
        Step:           ExprT
    }

    type State<'T> = {
        LastStep:       Tensor<'T>       
    }

    type StateExpr = {
        LastStep:       ExprT
    }

open GradientDescent

type GradientDescent<'T when 'T: equality and 'T: comparison> 
        (loss:   ExprT, pars:   ExprT, dev:    IDevice) =

    do Util.checkProperType<'T> ()
    do if loss.NDims <> 0 then failwith "loss must be a scalar"

    let cfg = {
        CfgExpr.Step        = Expr.var<'T> "GradientDescent.Cfg.Step" []
    }

    let state = {
        StateExpr.LastStep  = Expr.var<'T> "Adam.State.LastStep"    (Expr.shapeOf pars)
    }

    let rpCfg = VarRecord<Cfg<'T>, CfgExpr> (cfg, dev)
    let rpState = VarRecord<State<'T>, StateExpr> (state, dev)

    static member New loss pars dev =
        GradientDescent (loss, pars, dev) :> IOptimizer<'T, Cfg<'T>, State<'T>>

    static member DefaultCfg : Cfg<'T> = {
        Step        = conv<'T> 1e-4
    }

    member this.InitialState (cfg: Cfg<'T>) parVals : State<'T> = {
        LastStep    = HostTensor.zeros (ITensor.shape parVals) |> dev.ToDev
    }

    member this.Minimize =
        let grad = Deriv.compute loss |> Deriv.ofVar pars |> Expr.reshape (Expr.shapeOf pars)
        Expr.storeToVar pars (pars - cfg.Step * grad)

    member this.Use f =
        f |> rpState.Use |> rpCfg.Use

    member this.PublishLoc mb =
        rpCfg.PublishLocAndStride mb
        rpState.PublishLocAndStride mb

    interface IOptimizer<'T, Cfg<'T>, State<'T>> with
        member this.OptStepExpr = this.Minimize
        member this.Use f = this.Use f
        member this.CfgWithLearningRate learningRate cfg = {cfg with Step=conv<'T> learningRate}
        member this.InitialState cfg parVals = this.InitialState cfg parVals
        member this.LoadState hdf prefix = rpState.LoadValue hdf prefix
        member this.SaveState hdf prefix state = rpState.SaveValue hdf prefix state


