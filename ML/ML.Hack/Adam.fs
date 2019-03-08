namespace Optimizers

open DeepNet.Utils
open Tensor
open Tensor.Expr


module Adam =

    //type VarTensorExpr<'T> = {
    //    Var: Var<'T> option
    //    mutable Tensor: Tensor<'T> option
    //    Expr: Expr<'T>
    //} with 
    //    static member make (ctx, shp) =
    //        let var = Var<'T> (ctx, shp)
    //        {
    //            Var = Some var
    //            Tensor = None
    //            Expr = Expr.var var
    //        }

    //    member this.Inst (symSizes: SymSizeEnv) =
    //        match this.Var with
    //        | Some var ->
    //            let nShp = var.Shape |> ShapeSpec.substSymbols symSizes |> ShapeSpec.eval
    //            this.Tensor <- Some (Tensor<'T>.zeros var.Dev nShp)
    //        | None -> failwith "Can only instantiate when Var is present."    
            

    type Cfg = {
        StepData:           Data<float32>
        MomentumData:       Data<float32>
        DecayData:          Data<float32>
        DecayMom1Data:      Data<float32>
        DecayMom2Data:      Data<float32>
        OffsetData:         Data<float32>
    } with 
        static member make (ctx: Context) = {
            StepData      = Data<float32> (ctx / "Step",      Shape.scalar) 
            MomentumData  = Data<float32> (ctx / "Momentum",  Shape.scalar) 
            DecayData     = Data<float32> (ctx / "Decay",     Shape.scalar) 
            DecayMom1Data = Data<float32> (ctx / "DecayMom1", Shape.scalar)
            DecayMom2Data = Data<float32> (ctx / "DecayMom2", Shape.scalar) 
            OffsetData    = Data<float32> (ctx / "Offset",    Shape.scalar) 
        }

        member this.Step
            with get () = this.StepData.InstValue.Value
            and set value = this.StepData.InstValue.Value <- value
            
        member this.Momentum
            with get () = this.MomentumData.InstValue.Value
            and set value = this.MomentumData.InstValue.Value <- value

        member this.Decay
            with get () = this.DecayData.InstValue.Value
            and set value = this.DecayData.InstValue.Value <- value
            
        member this.DecayMom1
            with get () = this.DecayMom1Data.InstValue.Value
            and set value = this.DecayMom1Data.InstValue.Value <- value

        member this.DecayMom2
            with get () = this.DecayMom2Data.InstValue.Value
            and set value = this.DecayMom2Data.InstValue.Value <- value

        member this.Offset
            with get () = this.OffsetData.InstValue.Value
            and set value = this.OffsetData.InstValue.Value <- value


    type State<'T> = {
        Iter:           Data<'T>  
        LastStep:       Data<'T>
        EstMom1:        Data<'T>
        EstMom2:        Data<'T>
        EstMom1B:       Data<'T>
        EstMom2B:       Data<'T>
    } with
        static member make (ctx: Context) () =
            // how to handle multiple parameters and variables??
            // need a list of parameters to optimize wrt to
            // however, now this has become more difficult
            // we could instantiate multiple optimizers, one for each parameter
            // so only way would be to instantiate the optimizer for each parameter, correct?
            // yes, probably
            // also if the 
            // not  having one big parameter vector might kill us
            // substitution approch:
            // replace parameters with slices in big parameter vector
            // is not bad, but how will a model part access its numeric data if it wants to?
            // So, perhaps we need to introduce something like a DataRef?
            // It references a data value that can be freely allocated somewhere or
            // even replaced by another expression.
            // We can store such a marker in the graph, but how will the local reference be updated?
            // It would need to contain a write reference that is updated accordingly.
            // If we just could store the model building on the graph.
            ()

        


open Adam

type Adam<'T when 'T: equality and 'T: comparison> 
        (loss:  Expr, pars:  Expr,  cfg: CfgExpr,  state: StateExpr) =

    do Util.checkProperType<'T> ()
    do if loss.NDims <> 0 then failwith "loss must be a scalar"

    let cfg = {
        CfgExpr.Step        = Expr.var<'T> "Adam.Cfg.Step"          []
        CfgExpr.Momentum    = Expr.var<'T> "Adam.Cfg.Momentum"      []
        CfgExpr.Decay       = Expr.var<'T> "Adam.Cfg.Decay"         []
        CfgExpr.DecayMom1   = Expr.var<'T> "Adam.Cfg.DecayMom1"     []
        CfgExpr.DecayMom2   = Expr.var<'T> "Adam.Cfg.DecayMom2"     []
        CfgExpr.Offset      = Expr.var<'T> "Adam.Cfg.Offset"        []
    }

    let state = {
        StateExpr.Iter      = Expr.var<'T> "Adam.State.Iter"        []
        StateExpr.LastStep  = Expr.var<'T> "Adam.State.LastStep"    (Expr.shapeOf pars)
        StateExpr.EstMom1   = Expr.var<'T> "Adam.State.EstMom1"     (Expr.shapeOf pars)
        StateExpr.EstMom2   = Expr.var<'T> "Adam.State.EstMom2"     (Expr.shapeOf pars)
        StateExpr.EstMom1B  = Expr.var<'T> "Adam.State.EstMom1B"    (Expr.shapeOf pars)
        StateExpr.EstMom2B  = Expr.var<'T> "Adam.State.EstMom2B"    (Expr.shapeOf pars)            
    }

    let rpCfg = VarRecord<Cfg<'T>, CfgExpr> (cfg, dev)
    let rpState = VarRecord<State<'T>, StateExpr> (state, dev)

    static member New loss pars dev =
        Adam (loss, pars, dev) :> IOptimizer<'T, Cfg<'T>, State<'T>>

    static member DefaultCfg : Cfg<'T> = {
        Step        = conv<'T> 2e-4
        Momentum    = conv<'T> 0.0
        Decay       = conv<'T> (1.0 - 1e-8)
        DecayMom1   = conv<'T> 1e-1
        DecayMom2   = conv<'T> 1e-3
        Offset      = conv<'T> 1e-8       
    }

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
