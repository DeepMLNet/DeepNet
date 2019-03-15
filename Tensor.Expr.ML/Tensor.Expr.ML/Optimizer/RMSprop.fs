namespace Tensor.Expr.ML.Opt

open DeepNet.Utils
open Tensor
open Tensor.Expr


/// RMSprop optimizer.
module RMSprop =

    /// RMSprop optimizer configuration.
    type Cfg = {
        Step:           float
        Offset:         float
    } with
        /// Standard optimizer configuration.
        static member standard = {
            Step        = 2e-4
            Offset      = 1e-8       
        }        

        interface IOptimizerCfg with
            member this.LearningRate = this.Step
            member this.SetLearningRate step = {this with Step=step} :> _

    /// RMSprop optimizer state.
    type State = {
        EstMomSq: ITensor
    } with 
        static member initial typ dev shp = {
            EstMomSq = ITensor.zeros typ dev shp    
        }

        interface IOptimizerStatePart with      
            member this.Initial () =
                State.initial this.EstMomSq.DataType this.EstMomSq.Dev this.EstMomSq.Shape :> _ 
            member this.Data 
                with get () = 
                    Map [
                        "EstMomSq", this.EstMomSq
                    ]       

    type internal RMSpropPart (pars: Var, wrtParts: UExpr) =
        let typ = pars.DataType
        let dev = pars.Dev
        let shp = Shape.eval pars.Shape 
        let state = State.initial typ dev shp

        // configuration
        let step = ITensor.zeros typ dev []
        let offset = ITensor.zeros typ dev []

        interface IOptimizerPart<Cfg, State> with

            member this.Cfg 
                with set cfg =
                    step.Value <- convTo typ cfg.Step
                    offset.Value <- convTo typ cfg.Offset

            member this.State 
                with get () = IOptimizerStatePart.copy state
                and set value = IOptimizerStatePart.transferFrom value state 

            member this.Step =
                let gradient = wrtParts |> UExpr.reshape pars.Shape

                let oneHalf         = UExpr.scalar dev (convTo typ 0.5)
                let two             = UExpr.scalar dev (convTo typ 2)
                let onePointNine    = UExpr.scalar dev (convTo typ 0.9)
                let onePointOne     = UExpr.scalar dev (convTo typ 0.1)

                let estMomSq = onePointOne * gradient ** two + onePointNine * UExpr state.EstMomSq
                let step = UExpr step * gradient / (estMomSq ** oneHalf + UExpr offset)
                let newPars = UExpr pars - step

                EvalUpdateBundle.empty
                |> EvalUpdateBundle.addVar pars newPars
                |> EvalUpdateBundle.addData state.EstMomSq estMomSq


/// RMSprop optimizer.
type RMSprop =

    static member private makePart parts wrtParts =
        RMSprop.RMSpropPart (parts, wrtParts) :> IOptimizerPart<_, _>

    /// Create optimizer for the specified loss and parameter set instance.
    static member make (cfg, loss, parSetInst) =
        Optimizer (RMSprop.makePart, cfg, loss, parSetInst)

    /// Create optimizer for the specified loss and parameter set instance.
    static member make (cfg, loss: Expr<'T>, parSetInst) =
        Optimizer (RMSprop.makePart, cfg, loss.Untyped, parSetInst)
