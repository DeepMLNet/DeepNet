namespace Tensor.Expr.ML.Opt

open DeepNet.Utils
open Tensor
open Tensor.Backend
open Tensor.Expr


/// The state of an optimizer.
type IOptimizerState =
    /// All parts (for a particular data type and device) making up the state of the optimizer.
    abstract Parts: Map<TypeName * ITensorDevice, IOptimizerStatePart> with get
    /// All data making up the state of the optimizer.
    abstract Data: Map<TypeName * ITensorDevice, Map<string, ITensor>> with get
    /// A new state with initial state values.
    abstract Initial: unit -> IOptimizerState  
    /// Load state from a HDF5 file.
    abstract Load: hdf:HDF5 -> prefix:string -> unit
    /// Save state to a HDF5 file.
    abstract Save: hdf:HDF5 -> prefix:string -> unit

/// The configuration of an optimizer.
and IOptimizerCfg =
    /// Learning rate.
    abstract LearningRate: float with get
    /// Returns a new configuration with the specified learning rate.
    abstract SetLearningRate: float -> IOptimizerCfg

/// Optimizer.
and IOptimizer =
    /// Expression performing one optimization step.
    abstract Step: EvalUpdateBundle
    /// The optimizer configuration.
    abstract Cfg: IOptimizerCfg with get, set
    /// The optimizer state.
    abstract State: IOptimizerState with get, set

/// A part (for a particular data type and device) of an optimizer state.
and IOptimizerStatePart =
    /// All data making up the state.
    abstract Data: Map<string, ITensor> with get
    /// A new state with initial state values.
    abstract Initial: unit -> IOptimizerStatePart    

/// A part (for a particular data type and device) of an optimizer state.
module IOptimizerStatePart =
    /// Transfer `src` into `dst` state part.
    let transferFrom (src: 'State when 'State :> IOptimizerStatePart) (dst: 'State) =
        let srcData = src.Data
        let dstData = dst.Data
        for KeyValue(key, srcValue) in srcData do
            let dstValue = dstData.[key]
            dstValue.TransferFrom srcValue

    /// Return a copy of the state part.
    let copy (src: 'State when 'State :> IOptimizerStatePart) =
        let clone = src.Initial() :?> 'State
        transferFrom src clone
        clone

/// The state of an optimizer.
module IOptimizerState =
    /// Transfer `src` into `dst` state.
    let transferFrom (src: 'State when 'State :> IOptimizerState) (dst: 'State) =
        for KeyValue(key, srcPart) in src.Parts do
            IOptimizerStatePart.transferFrom srcPart dst.Parts.[key]

    /// Return a copy of the state part.
    let copy (src: 'State when 'State :> IOptimizerState) =
        let clone = src.Initial() :?> 'State
        transferFrom src clone
        clone

/// A part (for a particular data type and device) of an optimizer.
type IOptimizerPart<'Cfg, 'State when 'State :> IOptimizerStatePart> =
    abstract Cfg: 'Cfg with set
    abstract State: 'State with get, set
    abstract Step: EvalUpdateBundle 

/// The state of an optimizer.
type OptimizerState<'SP when 'SP :> IOptimizerStatePart> =
    | OptimizerState of Map<TypeName * ITensorDevice, 'SP>

    /// Iterate over all data contained in the optimizer state.
    static member iterData (fn: TypeName -> ITensorDevice -> string -> ITensor -> unit) (os: OptimizerState<'SP>) =
        for KeyValue((typ, dev), part) in (os :> IOptimizerState).Data do
            for KeyValue(name, value) in part do
                fn typ dev name value

    /// All parts (for a particular data type and device) making up the state of the optimizer.
    member this.Parts =
        let (OptimizerState state) = this
        state

    /// A new state with initial state values.
    member this.Initial () =
        this.Parts |> Map.map (fun _ part -> part.Initial() :?> 'SP) |> OptimizerState

    interface IOptimizerState with
        member this.Parts =
            this.Parts |> Map.map (fun _ part -> part :> IOptimizerStatePart)

        member this.Data =
            this.Parts |> Map.map (fun _ part -> part.Data)

        member this.Initial () =
            this.Initial () :> _ 

        member this.Load hdf prefix =
            this |> OptimizerState.iterData (fun typ dev name value ->
                let path = sprintf "%s/%A/%A/%s" prefix typ dev name 
                let data = HostTensor.readUntyped hdf path
                value.TransferFrom data)             

        member this.Save hdf prefix =
            this |> OptimizerState.iterData (fun typ dev name value ->
                let path = sprintf "%s/%A/%A/%s" prefix typ dev name 
                let data = ITensor.transfer HostTensor.Dev value
                HostTensor.write hdf path data)


/// Optimizer.
type Optimizer<'Cfg, 'SP when 'Cfg :> IOptimizerCfg and 'SP :> IOptimizerStatePart> 
        (createOpt: Var -> UExpr -> IOptimizerPart<'Cfg, 'SP>, 
         cfg: 'Cfg, 
         loss: UExpr, 
         parSetInst: ParSetInst) =

    do if loss.NDims <> 0 then 
        failwithf "Loss must be a scalar, but it has shape %A." loss.Shape

    let mutable cfg = cfg
    let deriv = Deriv.compute loss
    let parts =
        parSetInst.TypeDeviceVars
        |> Map.map (fun _ parPartVar -> 
            let part = createOpt parPartVar (deriv.Wrt parPartVar)
            part.Cfg <- cfg
            part)

    /// Optimizer configuration.
    member this.Cfg
        with get () = cfg
        and set newCfg =
            cfg <- newCfg
            for KeyValue(_, part) in parts do
                part.Cfg <- cfg

    /// Optimizer state.
    member this.State
        with get () =
            parts |> Map.map (fun _ part -> part.State) |> OptimizerState
        and set (OptimizerState value) =
            parts |> Map.iter (fun key part -> part.State <- value.[key])

    /// Update bundle for performing one optimization step.
    member this.Step =
        parts
        |> Map.toSeq
        |> Seq.map (fun (_, part) -> part.Step)
        |> EvalUpdateBundle.mergeMany

    interface IOptimizer with
        member this.Step = this.Step
        member this.Cfg 
            with get () = this.Cfg :> _
            and set newCfg = this.Cfg <- newCfg :?> 'Cfg
        member this.State 
            with get () = this.State :> _
            and set newState = this.State <- newState :?> OptimizerState<'SP>
            