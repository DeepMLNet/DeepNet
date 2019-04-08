namespace Tensor.Expr.Opt

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops
open Tensor.Expr.Opt.Tools



/// Optimizes loop expressions.
[<Optimizer>]
type LoopOptimizer() =

    interface IOptimizer with
        member __.Order = 60

    interface IMultiChannelOptimizer with
        member __.Optimize subOpt expr =
            match expr with
            | MultiChannelExpr.Loop loopSpec ->
                let optChannels = 
                    loopSpec.Channels                        
                    |> Map.map (fun _ch lv -> {lv with Expr=lv.Expr |> BaseExprCh.map subOpt})
                MultiChannelExpr {loopSpec with Channels=optChannels}

            | _ -> expr

