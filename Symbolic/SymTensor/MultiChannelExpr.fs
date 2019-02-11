namespace SymTensor

open SymTensor.Ops
open DeepNet.Utils



type MultiChannelExpr (op: IMultiChannelOp) =    
    inherit BaseMultiChannelExpr(op)

    new (baseExpr: BaseMultiChannelExpr) =
        MultiChannelExpr(baseExpr.Op)

    static member op (expr: MultiChannelExpr) = expr.Op
    static member typeNames (expr: MultiChannelExpr) = expr.TypeNames
    static member shapes (expr: MultiChannelExpr) = expr.Shapes
    static member nDims (expr: MultiChannelExpr) = expr.NDims
    static member nElems (expr: MultiChannelExpr) = expr.NElems
    static member vars (expr: MultiChannelExpr) = expr.Vars
    static member canEvalAllSymSizes (expr: MultiChannelExpr) = expr.CanEvalAllSymSizes
    static member substSymSizes (env: SymSizeEnv) (expr: MultiChannelExpr) =
        expr.SubstSymSizes env |> MultiChannelExpr

    member this.Item 
        with get (channel: string) = 
            {Channel.Channel=channel; X=this} |> MultiChannelExpr
