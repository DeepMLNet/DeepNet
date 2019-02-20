namespace Tensor.Expr.Compiler

open System
open System.Diagnostics
open System.Collections.Generic

open Tensor.Utils
open Tensor.Expr
open UExprTypes


[<AutoOpen>]
module ExecUnitsTypes = 

    /// a channel id
    type ChannelIdT = string

    /// manikins representing the data in each channel
    type ChannelManikinsT = Map<ChannelIdT, TensorManikin>

    /// manikins representing the data in each channel and flag if it is shared
    type ChannelManikinsAndSharedT = Map<ChannelIdT, TensorManikin * bool>

    /// requests for manikins representing the data in each channel
    type ChannelReqsT = Map<ChannelIdT, TensorManikin option>

    /// a command to be executed
    type IExecItem =
        abstract VisualizationText: string

    /// Id of an ExecUnitT
    type ExecUnitIdT = int

    /// a sequence of ExecItems to evaluate an op
    type ExecUnitT = {
        Id:           ExecUnitIdT 
        DependsOn:    ExecUnitIdT list
        Items:        IExecItem list
        Expr:         UExprT
        Manikins:     TensorManikin list
        Channels:     ChannelManikinsAndSharedT 
        Srcs:         TensorManikin list        
        ExtraMem:     StorageManikin list
        RerunAfter:   ExecUnitIdT list
    }

    /// result of an evaluation request
    type EvalResultT = {
        ExecUnitId:     ExecUnitIdT 
        Channels:       ChannelManikinsAndSharedT
    }

    /// an evaluation request
    type EvalReqT = {
        Id:             int 
        Expr:           UExprT
        ChannelReqs:    ChannelReqsT
        OnCompletion:   EvalResultT -> unit
    }

    type MemAllocatorT = TypeName -> int64 -> MemAllocKind -> StorageManikin

    type ExecItemsForOpArgs = {
        MemAllocator:       MemAllocatorT
        Target:             ChannelManikinsT
        Op:                 UOpT
        UExpr:              UExprT
        Metadata:           UMetadata
        Srcs:               ChannelManikinsAndSharedT list
        SubmitInitItems:    IExecItem list -> unit
    }

    type TraceItemsForExprArgs = {
        MemAllocator:       MemAllocatorT
        Target:             ChannelManikinsT
        Expr:               UExprT
    }

    type TrgtGivenSrcsArgs = {
        MemAllocator:       MemAllocatorT
        TargetRequest:      ChannelReqsT
        Op:                 UOpT
        Metadata:           UMetadata
        Srcs:               ChannelManikinsAndSharedT list    
    }

    type SrcReqsArgs = {
        TargetRequest:      ChannelReqsT
        Op:                 UOpT
        Metadata:           UMetadata
        SrcShapes:          Map<Channel, NShapeSpec> list 
    }

    /// record containing functions called by the ExecUnitT generator
    type ExecUnitsGeneratorT = {
        ExecItemsForOp:         ExecItemsForOpArgs -> IExecItem list
        TracePreItemsForExpr:   TraceItemsForExprArgs -> IExecItem list
        TracePostItemsForExpr:  TraceItemsForExprArgs -> IExecItem list
        TrgtGivenSrcs:          TrgtGivenSrcsArgs -> ChannelManikinsAndSharedT
        SrcReqs:                SrcReqsArgs -> ChannelReqsT list
    }

    /// generated ExecUnits for an expression
    type ExecUnitsForExprT = {
        Expr:           UExprT
        ExecUnits:      ExecUnitT list
        Result:         EvalResultT
        MemAllocs:      StorageAlloc list
        InitItems:      IExecItem list
    }

    /// Diagnostic information from a compilation.
    type CompileDiagnosticsT = {
        UExpr:              UExprT
        ExecUnits:          ExecUnitT list
        SubDiagnostics:     Map<UExprT, CompileDiagnosticsT>
    }
