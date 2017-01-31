namespace SymTensor

open Basics
open ArrayNDNS


/// functions for loop evaluation
module LoopEval =
    open Expr

    /// channel information for loop execution
    type LoopChannelInfoT = {
        Shape:      NShapeSpecT
        SliceDim:   int
        Target:     IArrayNDT
    }


    /// build strides information for loop sources and targets
    let buildStrides (vars: Map<VarSpecT, LoopInputT>) (args: IArrayNDT list) 
                     (channels: Map<ChannelT, LoopChannelInfoT>) 
                     : VarStridesT * ChannelStridesT * int list option list =

        let mutable argRequiredStrideOrder = List.replicate args.Length None

        let varStrides = 
            vars |> Map.map (fun vs li ->
                match li with
                | ConstArg idx -> 
                    args.[idx].Layout.Stride
                | SequenceArgSlice {ArgIdx=idx; SliceDim=dim} ->
                    args.[idx].Layout.Stride |> List.without dim
                | PreviousChannel {Channel=ch; InitialArg=ivIdx} ->
                    let sliceDim = channels.[ch].SliceDim
                    let chStride = channels.[ch].Target.Layout.Stride |> List.without sliceDim

                    // check that initial value has same stride as channel target
                    let ivStride = args.[ivIdx].Layout.Stride |> List.without sliceDim
                    if chStride <> ivStride then
                        // Stride mismatch. 
                        // Check that copying argument to temporary array would
                        // result in matching strides.
                        let shp = args.[ivIdx].Shape
                        let strideOrder = 
                            [0 .. shp.Length-1] |> List.swap 0 sliceDim |> List.rev
                        let ivCopyStride = 
                            ArrayNDLayout.orderedStride args.[ivIdx].Shape strideOrder
                            |> List.without sliceDim
                        if chStride <> ivCopyStride then 
                            printfn "Loop stride problem:"
                            printfn "Channel %s:\n%A" ch channels.[ch]
                            printfn "Initial value layout:\n%A" args.[ivIdx].Layout
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
            channels |> Map.map (fun ch lv -> lv.Target.Layout.Stride |> List.without lv.SliceDim)

        varStrides, channelStrides, argRequiredStrideOrder

    /// builds inputs and outputs for one loop iteration 
    let buildInOut (iter: int64) (iterAry: IArrayNDT) (itersRemainingAry: IArrayNDT)
                   (vars: Map<VarSpecT, LoopInputT>)
                   (args: IArrayNDT list) (channels: Map<ChannelT, LoopChannelInfoT>)
                   : VarEnvT * Map<ChannelT, IArrayNDT> =

        /// RngAll in all dimensions but specified one
        let rngAllBut ary dim dimSlice = 
            List.replicate (ArrayND.nDims ary) RngAll
            |> List.set dim dimSlice

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
                        let slice = rngAllBut args.[idx] dim (RngElem iter)
                        args.[idx].[slice] 
                    | PreviousChannel {Channel=ch; Delay=delay; InitialArg=ivIdx} ->
                        let delay = SizeSpec.eval delay
                        let dim = channels.[ch].SliceDim
                        let prvIter = iter - delay
                        if prvIter >= 0L then
                            let slice = rngAllBut channels.[ch].Target dim (RngElem prvIter)
                            channels.[ch].Target.[slice]
                        else
                            let initialIter = args.[ivIdx].Shape.[dim] + prvIter
                            let slice = rngAllBut args.[ivIdx] dim (RngElem initialIter)
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
            channels |> Map.map (fun ch lci ->
                let slice = rngAllBut lci.Target lci.SliceDim (RngElem iter)
                lci.Target.[slice])

        srcVarEnv, targets

