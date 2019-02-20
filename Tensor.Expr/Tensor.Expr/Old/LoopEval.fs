namespace Tensor.Expr

open Tensor
open Tensor.Backend
open DeepNet.Utils


/// functions for loop evaluation
module LoopEval =
    open Expr

    /// channel information for loop execution
    type LoopChannelInfoT = {
        Shape:          NShapeSpec
        SliceDim:       int
        Target:         ITensor
    } 

    /// channel layout information for building loop strides
    type LoopChannelLayoutInfoT = {
        Shape:          NShapeSpec
        SliceDim:       int
        TargetLayout:   TensorLayout
    } 

    /// build strides information for loop sources and targets
    let buildStrides (vars: Map<Var, LoopInput>) (args: TensorLayout list) 
                     (channels: Map<Channel, LoopChannelLayoutInfoT>) 
                     : VarStrides * ChannelStridesT * int list option list =

        let mutable argRequiredStrideOrder = List.replicate args.Length None

        let varStrides = 
            vars |> Map.map (fun vs li ->
                match li with
                | ConstArg idx -> 
                    args.[idx].Stride
                | SequenceArgSlice {ArgIdx=idx; SliceDim=dim} ->
                    args.[idx].Stride |> List.without dim
                | PreviousChannel {Channel=ch; InitialArg=ivIdx} ->
                    let sliceDim = channels.[ch].SliceDim
                    let chStride = channels.[ch].TargetLayout.Stride |> List.without sliceDim

                    // check that initial value has same stride as channel target
                    let ivStride = args.[ivIdx].Stride |> List.without sliceDim
                    if chStride <> ivStride then
                        // Stride mismatch. 
                        // Check that copying argument to temporary array would
                        // result in matching strides.
                        let shp = args.[ivIdx].Shape
                        let strideOrder = 
                            [0 .. shp.Length-1] |> List.swap 0 sliceDim |> List.rev
                        let ivCopyStride = 
                            TensorLayout.orderedStride args.[ivIdx].Shape strideOrder
                            |> List.without sliceDim
                        if chStride <> ivCopyStride then 
                            printfn "Loop stride problem:"
                            printfn "Channel %s:\n%A" ch channels.[ch]
                            printfn "Initial value layout:\n%A" args.[ivIdx]
                            printfn "Copy stride:    %A" ivCopyStride
                            printfn "Channel stride: %A" chStride
                            failwithf "channel %A slice strides %A are different from initial \
                                       value slice strides %A for loop variable %A and copying \
                                       to a temporary array would not help" 
                                      ch chStride ivStride vs
                        // request copy to C-strided temporary array
                        argRequiredStrideOrder <- 
                            argRequiredStrideOrder |> List.set ivIdx (Some strideOrder)
                    chStride
                | IterationIndex
                | IterationsRemaining -> [])

        let channelStrides =
            channels |> Map.map (fun ch lv -> lv.TargetLayout.Stride |> List.without lv.SliceDim)

        varStrides, channelStrides, argRequiredStrideOrder

    /// builds inputs and outputs for one loop iteration 
    let buildInOut (nIters: int64) (iter: int64) (iterAry: ITensor) (itersRemainingAry: ITensor)
                   (vars: Map<Var, LoopInput>)
                   (args: ITensor list) (channels: Map<Channel, LoopChannelInfoT>)
                   : VarEnv * Map<Channel, ITensor> =

        /// RngAll in all dimensions but specified one
        let rngAllBut ary dim dimSlice = 
            List.replicate (ITensor.nDims ary) Rng.All
            |> List.set dim dimSlice

        /// The slice of the channel's target for the specified iteration.
        let targetSlice ch iter =
            let dim = channels.[ch].SliceDim
            let trgtSize = channels.[ch].Target.Shape.[dim]
            // calculate offset so that last loop iteration is written into last element of
            // channel's target
            let offset = trgtSize - 1L - ((nIters - 1L) % trgtSize)
            assert ((offset + nIters - 1L) % trgtSize = trgtSize - 1L)
            let pos = (offset + iter) % trgtSize
            let slice = rngAllBut channels.[ch].Target dim (Rng.Elem pos)
            channels.[ch].Target.[slice]

        // build variable environment for value sources
        let srcVarEnv = 
            vars
            |> Map.map (fun vs li ->
                // get value for variable
                let value = 
                    match li with
                    | ConstArg idx -> 
                        args.[idx] 
                    | SequenceArgSlice {ArgIdx=idx; SliceDim=dim} -> 
                        let slice = rngAllBut args.[idx] dim (Rng.Elem iter)
                        args.[idx].[slice] 
                    | PreviousChannel {Channel=ch; Delay=delay; InitialArg=ivIdx} ->
                        let delay = SizeSpec.eval delay
                        let dim = channels.[ch].SliceDim
                        if channels.[ch].Target.Shape.[dim] < delay then
                            failwithf "target for channel %A has insufficient size %d for delay %d"
                                       ch channels.[ch].Target.Shape.[dim] delay
                        let prvIter = iter - delay
                        if prvIter >= 0L then targetSlice ch prvIter
                        else
                            let initialIter = args.[ivIdx].Shape.[dim] + prvIter
                            let slice = rngAllBut args.[ivIdx] dim (Rng.Elem initialIter)
                            args.[ivIdx].[slice] 
                    | IterationIndex -> iterAry
                    | IterationsRemaining -> itersRemainingAry

                // check type and shape
                if ShapeSpec.eval vs.Shape <> value.Shape then
                    failwithf "loop variable %A got value with shape %A" vs value.Shape
                if vs.Type <> value.DataType then
                    failwithf "loop variable %A got value with data type %A" vs value.DataType
                    
                value)

        // slice outputs into channel targets
        let targets =
            channels |> Map.map (fun ch _ -> targetSlice ch iter)

        srcVarEnv, targets

