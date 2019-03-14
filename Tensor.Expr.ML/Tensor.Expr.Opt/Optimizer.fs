namespace Tensor.Expr.Opt

open Tensor.Utils
open Tensor
open Tensor.Expr


///// Optimizer.
//type IOptimizer<'T, 'OptCfg, 'OptState> =

//    /// Expression performing one optimization step.
//    abstract member OptStepExpr: Expr

//    abstract member Use: (VarEnv -> 'A) -> (VarEnv -> 'OptCfg -> 'OptState -> 'A)

//    /// Returns the configuration with the learning rate set to the specified value.
//    abstract member CfgWithLearningRate: learningRate:float -> cfg:'OptCfg -> 'OptCfg

//    /// Initial optimizer state.
//    abstract member InitialState: cfg:'OptCfg -> parameterValues:ITensor -> 'OptState

//    /// Loads the optimizer state from a HDF5 file with given prefix.
//    abstract member LoadState: hdf:HDF5 -> prefix:string -> 'OptState

//    /// Saves the optimizer state to a HDF5 file with given prefix.
//    abstract member SaveState: hdf:HDF5 -> prefix:string -> state:'OptState -> unit


        
type IOptimizerPart<'Cfg> =
    abstract Cfg: 'Cfg with set
    abstract Step: EvalUpdateBundle with get


/// Optimizer.
type Optimizer<'Cfg> (createOpt: Var -> UExpr -> IOptimizerPart<'Cfg>, 
                      cfg: 'Cfg, 
                      loss: UExpr, 
                      parSetInst: ParSetInst) =

    do if loss.NDims <> 0 then 
        failwithf "Loss must be a scalar, but it has shape %A." loss.Shape

    let mutable cfg = cfg

    let deriv = Deriv.compute loss

    let parts =
        parSetInst.TypeDeviceVars
        |> Map.toSeq
        |> Seq.map (fun (_, parPartVar) -> 
            let part = createOpt parPartVar (deriv.Wrt parPartVar)
            part.Cfg <- cfg
            part)
        |> List.ofSeq


    /// Optimizer configuration.
    member this.Cfg
        with get () = cfg
        and set newCfg =
            cfg <- newCfg
            for part in parts do
                part.Cfg <- cfg

    /// Update bundle for performing one optimization step.
    member this.Step =
        parts
        |> Seq.map (fun part -> part.Step)
        |> EvalUpdateBundle.mergeMany

//    //interface IOptimizer<'T, Cfg<'T>, State<'T>> with
//    //    member this.OptStepExpr = this.Minimize
//    //    member this.Use f = this.Use f
//    //    member this.CfgWithLearningRate learningRate cfg = {cfg with Step=conv<'T> learningRate}
//    //    member this.InitialState cfg parVals = this.InitialState cfg parVals
//    //    member this.LoadState hdf prefix = rpState.LoadValue hdf prefix
//    //    member this.SaveState hdf prefix state = rpState.SaveValue hdf prefix state
