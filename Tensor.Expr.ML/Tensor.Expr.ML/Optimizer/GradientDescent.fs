namespace Tensor.Expr.ML.Opt

open DeepNet.Utils
open Tensor
open Tensor.Expr


/// Gradient descent optimizer.
module GradientDescent = 
    // TODO: momentum 

    /// Gradient descent optimizer configuration.
    type Cfg = {
        /// Step size.
        Step:           float
    } with
        /// Standard optimizer configuration.
        static member standard = {
            Step = 1e-5
        }

        interface IOptimizerCfg with
            member this.LearningRate = this.Step
            member this.SetLearningRate step = {this with Step=step} :> _

    /// Gradient descent optimizer state.
    type State = {
        LastStep: ITensor
    } with 
        static member initial typ dev shp = {
            LastStep = ITensor.zeros typ dev shp    
        }

        interface IOptimizerStatePart with      
            member this.Initial () =
                State.initial this.LastStep.DataType this.LastStep.Dev this.LastStep.Shape :> _ 
            member this.Data 
                with get () = 
                    Map [
                        "LastStep", this.LastStep
                    ]       

    type internal GradientDescentPart (pars: Var, wrtParts: UExpr) =
        let typ = pars.DataType
        let dev = pars.Dev
        let shp = Shape.eval pars.Shape 
        let state = State.initial typ dev shp

        // configuration
        let step = ITensor.zeros typ dev []

        interface IOptimizerPart<Cfg, State> with

            member this.Cfg 
                with set cfg =
                    step.Value <- convTo typ cfg.Step

            member this.State 
                with get () = IOptimizerStatePart.copy state
                and set value = IOptimizerStatePart.transferFrom value state 

            member this.Step =
                let gradient = wrtParts |> UExpr.reshape pars.Shape
                let newPars = UExpr pars - UExpr step * gradient

                EvalUpdateBundle.empty
                |> EvalUpdateBundle.addVar pars newPars



/// Gradient descent optimizer.
type GradientDescent =

    static member private makePart parts wrtParts =
        GradientDescent.GradientDescentPart (parts, wrtParts) :> IOptimizerPart<_, _>

    /// Create optimizer for the specified loss and parameter set instance.
    static member make (cfg, loss, parSetInst) =
        Optimizer (GradientDescent.makePart, cfg, loss, parSetInst)

    /// Create optimizer for the specified loss and parameter set instance.
    static member make (cfg, loss: Expr<'T>, parSetInst) =
        Optimizer (GradientDescent.makePart, cfg, loss.Untyped, parSetInst)

