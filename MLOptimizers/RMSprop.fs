namespace Optimizers

open Basics
open ArrayNDNS
open SymTensor


module RMSprop =

    type Cfg<'T> = {
        Step:           'T
        Offset:         'T
    } 

    type CfgExpr = {
        Step:           ExprT
        Offset:         ExprT
    }

    type State<'T> = {
        EstMom2:        ArrayNDT<'T>
    } 

    type StateExpr = {
        EstMom2:        ExprT
    }

open RMSprop

type RMSprop<'T when 'T: equality and 'T: comparison> 
        (loss:  ExprT, pars:  ExprT, dev:   IDevice) =

    do Util.checkProperType<'T> ()
    do if loss.NDims <> 0 then failwith "loss must be a scalar"

    let cfg = {
        CfgExpr.Step        = Expr.var<'T> "RMSprop.Cfg.Step"          []
        CfgExpr.Offset      = Expr.var<'T> "RMSprop.Cfg.Offset"        []
    }

    let state = {
        StateExpr.EstMom2   = Expr.var<'T> "RMSprop.State.EstMom2"     (Expr.shapeOf pars)          
    }

    let rpCfg = VarRecord<Cfg<'T>, CfgExpr> (cfg, dev)
    let rpState = VarRecord<State<'T>, StateExpr> (state, dev)

    static member New loss pars dev =
        RMSprop (loss, pars, dev) :> IOptimizer<'T, Cfg<'T>, State<'T>>

    static member DefaultCfg : Cfg<'T> = {
        Step        = conv<'T> 2e-4
        Offset      = conv<'T> 1e-8       
    }

    member this.InitialState (cfg: Cfg<'T>) parVals : State<'T> =
        let shp = ArrayND.shape parVals
        {
            EstMom2     = ArrayNDHost.zeros shp |> dev.ToDev
        }

    member this.Minimize : ExprT =
        let gradient = Deriv.compute loss |> Deriv.ofVar pars |> Expr.reshape (Expr.shapeOf pars) 
        //let gradient = gradient |> Expr.checkFinite "gradient"

        let oneHalf         = Expr.scalarOfSameType loss 0.5
        let two             = Expr.scalarOfSameType loss 2
        let onePointNine    = Expr.scalarOfSameType loss 0.9
        let onePointOne     = Expr.scalarOfSameType loss 0.1

        let o = cfg.Offset


        let estMom2 = onePointOne * gradient ** two + onePointNine * state.EstMom2
        let step = cfg.Step * gradient / (estMom2 ** oneHalf + o)
           
        Expr.discard [
            Expr.storeToVar pars (pars - step)
            Expr.storeToVar state.EstMom2 estMom2
        ]            

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
        member this.LoadState path = rpState.LoadValue path
        member this.SaveState path state = rpState.SaveValue path state
