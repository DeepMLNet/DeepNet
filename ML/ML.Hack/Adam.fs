namespace Optimizers.Adam

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Backend



type Cfg = {
    Step:           float
    Momentum:       float
    Decay:          float
    DecayMom1:      float
    DecayMom2:      float
    Offset:         float
} with 
    static member standard = {
        Step        = 2e-4
        Momentum    = 0.0
        Decay       = (1.0 - 1e-8)
        DecayMom1   = 1e-1
        DecayMom2   = 1e-3
        Offset      = 1e-8       
    }


type State = {
    Iter:           ITensor
    LastStep:       ITensor
    EstMom1:        ITensor
    EstMom2:        ITensor
    EstMom1B:       ITensor
    EstMom2B:       ITensor
} with
    static member make dataType dev shape = {
        Iter = ITensor.zeros dataType dev []
        LastStep = ITensor.zeros dataType dev shape
        EstMom1 = ITensor.zeros dataType dev shape
        EstMom2 = ITensor.zeros dataType dev shape
        EstMom1B = ITensor.zeros dataType dev shape
        EstMom2B = ITensor.zeros dataType dev shape
    }
            

type AdamPart (grad: UExpr, value: UExpr) =
    let typ = value.DataType
    let dev = value.Dev
    let shp = value.Shape
    let symShp = ShapeSpec.fix shp

    let one = UExpr.scalar dev (convTo typ 1.0)
    let oneHalf = UExpr.scalar dev (convTo typ 1.5)
    let two = UExpr.scalar dev (convTo typ 2.0)

    let step = ITensor.zeros typ dev []
    let momentum = ITensor.zeros typ dev []
    let decay = ITensor.zeros typ dev []
    let decayMom1 = ITensor.zeros typ dev []
    let decayMom2 = ITensor.zeros typ dev []
    let offset = ITensor.zeros typ dev []

    let iter = ITensor.zeros typ dev []
    let lastStep = ITensor.zeros typ dev symShp
    let _estMom1 = ITensor.zeros typ dev symShp
    let _estMom2 = ITensor.zeros typ dev symShp
    let _estMom1B = ITensor.zeros typ dev symShp
    let _estMom2B = ITensor.zeros typ dev symShp

    member this.ApplyCfg (cfg: Cfg) =
        step.Value <- convTo typ cfg.Step
        momentum.Value <- convTo typ cfg.Momentum
        decay.Value <- convTo typ cfg.Decay
        decayMom1.Value <- convTo typ cfg.DecayMom1
        decayMom2.Value <- convTo typ cfg.DecayMom2
        offset.Value <- convTo typ cfg.Offset

    member this.Step =
        let gradient = grad |> UExpr.reshape value.Shape
        //let gradient = gradient |> Expr.checkFinite "gradient"

        let m, d, o = UExpr momentum, UExpr decay, UExpr offset
        let dm1, dm2 = UExpr decayMom1, UExpr decayMom2
        let t = UExpr iter + one

        let coeff1 = one - (one - dm1) * d ** (t - one)
        let estMom1B = coeff1 * gradient + (one - coeff1) * _estMom1B
        let estMom2B = dm2 * gradient ** two + (one - dm2) * _estMom2B
        let estMom1 = _estMom1B / (one - (one - dm1) ** t + o)
        let estMom2 = _estMom2B / (one - (one - dm2) ** t + o)

        let step1 = step * estMom1 / (estMom2 ** oneHalf + o)
        let step2 = m * lastStep
        let step = step1 + step2
        let newValue = value - step

        // so need another update bundle here to update state?
        // or make this an imperative method?
        // but then, how to eval something and update at the same time?

           
        Expr.discard [
            Expr.storeToVar pars (pars - step)
            Expr.storeToVar state.Iter (state.Iter + one)
            Expr.storeToVar state.LastStep step
            Expr.storeToVar state.EstMom1 estMom1
            Expr.storeToVar state.EstMom2 estMom2
            Expr.storeToVar state.EstMom1B estMom1B
            Expr.storeToVar state.EstMom2B estMom2B
        ]            


type Adam (loss: UExpr, parSetInst: ParSetInst) =

    do if loss.NDims <> 0 then 
        failwithf "Loss must be a scalar, but it has shape %A." loss.Shape

    let parts =
        parSetInst.TypeDeviceValues
        |> Map.toSeq
        |> Seq.map (fun (_, value) -> State.make value.DataType value.Dev value.Shape)
        |> List.ofSeq

    let cfgs = 
        parSetInst.Ty
    member val Cfg = Cfg.make 


    member this.InitialState (cfg: Cfg<'T>) parVals : State<'T> =
        let shp = ITensor.shape parVals
        {
            Iter        = HostTensor.zeros []  |> dev.ToDev
            LastStep    = HostTensor.zeros shp |> dev.ToDev
            EstMom1     = HostTensor.zeros shp |> dev.ToDev
            EstMom2     = HostTensor.zeros shp |> dev.ToDev
            EstMom1B    = HostTensor.zeros shp |> dev.ToDev
            EstMom2B    = HostTensor.zeros shp |> dev.ToDev
        }

    member this.Minimize : Expr =
        let gradient = Deriv.compute loss |> Deriv.ofVar pars |> Expr.reshape (Expr.shapeOf pars) 
        //let gradient = gradient |> Expr.checkFinite "gradient"

        let one = Expr.scalarOfSameType loss 1
        let oneHalf = Expr.scalarOfSameType loss 0.5
        let two = Expr.scalarOfSameType loss 2

        let m, d, o = cfg.Momentum, cfg.Decay, cfg.Offset
        let dm1, dm2 = cfg.DecayMom1, cfg.DecayMom2
        let t = state.Iter + one

        let coeff1 = one - (one - dm1) * d ** (t - one)
        let estMom1B = coeff1 * gradient + (one - coeff1) * state.EstMom1B
        let estMom2B = dm2 * gradient ** two + (one - dm2) * state.EstMom2B
        let estMom1 = estMom1B / (one - (one - dm1) ** t + o)
        let estMom2 = estMom2B / (one - (one - dm2) ** t + o)

        let step1 = cfg.Step * estMom1 / (estMom2 ** oneHalf + o)
        let step2 = m * state.LastStep
        let step = step1 + step2
           
        Expr.discard [
            Expr.storeToVar pars (pars - step)
            Expr.storeToVar state.Iter (state.Iter + one)
            Expr.storeToVar state.LastStep step
            Expr.storeToVar state.EstMom1 estMom1
            Expr.storeToVar state.EstMom2 estMom2
            Expr.storeToVar state.EstMom1B estMom1B
            Expr.storeToVar state.EstMom2B estMom2B
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
        member this.LoadState hdf prefix = rpState.LoadValue hdf prefix
        member this.SaveState hdf prefix state = rpState.SaveValue hdf prefix state
