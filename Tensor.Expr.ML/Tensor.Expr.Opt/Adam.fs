namespace Tensor.Expr.Opt

open DeepNet.Utils
open Tensor
open Tensor.Expr


/// Adam optimizer.
module Adam =

    /// Adam optimizer configuration.
    type Cfg = {
        /// Step size.
        Step:           float
        /// Momentum.
        Momentum:       float
        /// Decay.
        Decay:          float
        ///
        DecayMom1:      float
        ///
        DecayMom2:      float
        ///
        Offset:         float
    } with 
        /// Standard optimizer configuration.
        static member standard = {
            Step        = 2e-4
            Momentum    = 0.0
            Decay       = (1.0 - 1e-8)
            DecayMom1   = 1e-1
            DecayMom2   = 1e-3
            Offset      = 1e-8       
        }


    type internal AdamPart (pars: Var, wrtParts: UExpr) =
        let typ = pars.DataType
        let dev = pars.Dev
        let shp = Shape.eval pars.Shape 

        // configuration
        let step = ITensor.zeros typ dev []
        let momentum = ITensor.zeros typ dev []
        let decay = ITensor.zeros typ dev []
        let decayMom1 = ITensor.zeros typ dev []
        let decayMom2 = ITensor.zeros typ dev []
        let offset = ITensor.zeros typ dev []

        // state
        let _iter = ITensor.zeros typ dev []
        let _lastStep = ITensor.zeros typ dev shp
        let _estMom1 = ITensor.zeros typ dev shp
        let _estMom2 = ITensor.zeros typ dev shp
        let _estMom1B = ITensor.zeros typ dev shp
        let _estMom2B = ITensor.zeros typ dev shp

        interface IOptimizerPart<Cfg> with

            member this.Cfg 
                with set cfg =
                    step.Value <- convTo typ cfg.Step
                    momentum.Value <- convTo typ cfg.Momentum
                    decay.Value <- convTo typ cfg.Decay
                    decayMom1.Value <- convTo typ cfg.DecayMom1
                    decayMom2.Value <- convTo typ cfg.DecayMom2
                    offset.Value <- convTo typ cfg.Offset

            member this.Step =
                let gradient = wrtParts |> UExpr.reshape pars.Shape
                //let gradient = gradient |> Expr.checkFinite "gradient"

                let one = UExpr.scalar dev (convTo typ 1.0)
                let oneHalf = UExpr.scalar dev (convTo typ 1.5)
                let two = UExpr.scalar dev (convTo typ 2.0)

                let m, d, o = UExpr momentum, UExpr decay, UExpr offset
                let dm1, dm2 = UExpr decayMom1, UExpr decayMom2
                let t = UExpr _iter + one

                let coeff1 = one - (one - dm1) * d ** (t - one)
                let estMom1B = coeff1 * gradient + (one - coeff1) * UExpr _estMom1B
                let estMom2B = dm2 * gradient ** two + (one - dm2) * UExpr _estMom2B
                let estMom1 = estMom1B / (one - (one - dm1) ** t + o)
                let estMom2 = estMom2B / (one - (one - dm2) ** t + o)

                let step1 = UExpr step * estMom1 / (estMom2 ** oneHalf + o)
                let step2 = m * UExpr _lastStep
                let step = step1 + step2
                let newPars = UExpr pars - step

                EvalUpdateBundle.empty
                |> EvalUpdateBundle.addVar pars newPars
                |> EvalUpdateBundle.addData _iter t
                |> EvalUpdateBundle.addData _lastStep step
                |> EvalUpdateBundle.addData _estMom1 estMom1
                |> EvalUpdateBundle.addData _estMom2 estMom2
                |> EvalUpdateBundle.addData _estMom1B estMom1B
                |> EvalUpdateBundle.addData _estMom2B estMom2B

    

/// Adam optimizer.
type Adam =

    static member private makePart parts wrtParts =
        Adam.AdamPart (parts, wrtParts) :> IOptimizerPart<_>

    /// Create optimizer for the specified loss and parameter set instance.
    static member make (cfg, loss, parSetInst) =
        Optimizer (Adam.makePart, cfg, loss, parSetInst)

    /// Create optimizer for the specified loss and parameter set instance.
    static member make (cfg, loss: Expr<'T>, parSetInst) =
        Optimizer (Adam.makePart, cfg, loss.Untyped, parSetInst)


