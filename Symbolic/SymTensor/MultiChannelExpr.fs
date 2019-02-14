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

    /// Accesses the specified channel of this multi-channel expression.
    member this.Item 
        with get (channel: string) = 
            {Channel.Channel=channel; X=this} |> Expr

    /// A loop provides iterative evaluation of one or multiple expresisons.
    /// All variables occurs in the loop channel expressions must be defined as loop variables.
    /// The function `loop` performs automatic lifting of constants and thus allows for easy
    /// usage of variables external to the loop.
    static member loopNoLift length vars channels (xs: Expr list) =
        let xs = xs |> List.map (fun x -> x :> BaseExpr)
        Ops.Loop.noLift length vars channels xs |> MultiChannelExpr

    /// A loop provides iterative evaluation of one or multiple expresisons.
    static member loop length vars channels (xs: Expr list) =
        let xs = xs |> List.map (fun x -> x :> BaseExpr)
        Ops.Loop.withLift length vars channels xs |> MultiChannelExpr