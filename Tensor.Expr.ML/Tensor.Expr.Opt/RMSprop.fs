namespace Tensor.Expr.Opt

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


    type internal RMSpropPart (pars: Var, wrtParts: UExpr) =
        let typ = pars.DataType
        let dev = pars.Dev
        let shp = Shape.eval pars.Shape 

        // configuration
        let step = ITensor.zeros typ dev []
        let offset = ITensor.zeros typ dev []

        // state
        let _estMomSq = ITensor.zeros typ dev shp

        interface IOptimizerPart<Cfg> with

            member this.Cfg 
                with set cfg =
                    step.Value <- convTo typ cfg.Step
                    offset.Value <- convTo typ cfg.Offset

            member this.Step =
                let gradient = wrtParts |> UExpr.reshape pars.Shape

                let oneHalf         = UExpr.scalar dev (convTo typ 0.5)
                let two             = UExpr.scalar dev (convTo typ 2)
                let onePointNine    = UExpr.scalar dev (convTo typ 0.9)
                let onePointOne     = UExpr.scalar dev (convTo typ 0.1)

                let estMomSq = onePointOne * gradient ** two + onePointNine * UExpr _estMomSq
                let step = UExpr step * gradient / (estMomSq ** oneHalf + UExpr offset)
                let newPars = UExpr pars - step

                EvalUpdateBundle.empty
                |> EvalUpdateBundle.addVar pars newPars
                |> EvalUpdateBundle.addData _estMomSq estMomSq


/// RMSprop optimizer.
type RMSprop =

    static member private makePart parts wrtParts =
        RMSprop.RMSpropPart (parts, wrtParts) :> IOptimizerPart<_>

    /// Create optimizer for the specified loss and parameter set instance.
    static member make (cfg, loss, parSetInst) =
        Optimizer (RMSprop.makePart, cfg, loss, parSetInst)

    /// Create optimizer for the specified loss and parameter set instance.
    static member make (cfg, loss: Expr<'T>, parSetInst) =
        Optimizer (RMSprop.makePart, cfg, loss.Untyped, parSetInst)
