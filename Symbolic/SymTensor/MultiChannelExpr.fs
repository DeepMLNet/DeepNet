namespace SymTensor

open SymTensor.Ops
open DeepNet.Utils



type MultiChannelExpr (baseExpr: BaseMultiChannelExpr) =    
    
    new (op: IMultiChannelOp) =
        MultiChannelExpr (BaseMultiChannelExpr.ofOp op)

    member this.BaseExpr = baseExpr
    static member baseExpr (expr: MultiChannelExpr) = expr.BaseExpr

    member this.Op = baseExpr.Op
    static member op (expr: MultiChannelExpr) = expr.Op

    member this.TypeNames = baseExpr.TypeNames
    static member typeNames (expr: MultiChannelExpr) = expr.TypeNames

    member this.Shapes = baseExpr.Shapes
    static member shapes (expr: MultiChannelExpr) = expr.Shapes

    member this.NDims = baseExpr.NDims
    static member nDims (expr: MultiChannelExpr) = expr.NDims

    member this.NElems = baseExpr.NElems
    static member nElems (expr: MultiChannelExpr) = expr.NElems

    member this.Vars = baseExpr.Vars
    static member vars (expr: MultiChannelExpr) = expr.Vars

    member this.CanEvalAllSymSizes = baseExpr.CanEvalAllSymSizes
    static member canEvalAllSymSizes (expr: MultiChannelExpr) = expr.CanEvalAllSymSizes
    
    static member substSymSizes (env: SymSizeEnv) (expr: MultiChannelExpr) =
        expr.BaseExpr |> BaseMultiChannelExpr.substSymSizes env |> MultiChannelExpr

    /// Accesses the specified channel of this multi-channel expression.
    member this.Item 
        with get (channel: string) = 
            {Channel.Channel=channel; X=this.BaseExpr} |> Expr

    /// A loop provides iterative evaluation of one or multiple expresisons.
    /// All variables occurs in the loop channel expressions must be defined as loop variables.
    /// The function `loop` performs automatic lifting of constants and thus allows for easy
    /// usage of variables external to the loop.
    static member loopNoLift length vars channels (xs: Expr list) =
        let xs = xs |> List.map Expr.baseExpr
        Ops.Loop.noLift length vars channels xs |> MultiChannelExpr

    /// A loop provides iterative evaluation of one or multiple expresisons.
    static member loop length vars channels (xs: Expr list) =
        let xs = xs |> List.map Expr.baseExpr
        Ops.Loop.withLift length vars channels xs |> MultiChannelExpr