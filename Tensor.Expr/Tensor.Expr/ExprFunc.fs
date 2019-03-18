namespace Tensor.Expr

open DeepNet.Utils
open Tensor


type ExprFunc private() =

    static member make (bndl: EvalUpdateBundle) =
        fun (varEnv: VarEnv) ->
            let vals = bndl |> EvalUpdateBundle.exec varEnv 
            vals     

    static member make (expr0: Expr<'T0>) =
        let bndl = 
            EvalUpdateBundle.empty
            |> EvalUpdateBundle.addExpr expr0
        fun (varEnv: VarEnv) ->
            let vals = bndl |> EvalUpdateBundle.exec varEnv 
            vals.Get expr0
            
    static member make (expr0: Expr<'T0>, expr1: Expr<'T1>) =
        let bndl = 
            EvalUpdateBundle.empty
            |> EvalUpdateBundle.addExpr expr0
            |> EvalUpdateBundle.addExpr expr1
        fun (varEnv: VarEnv) ->
            let vals = bndl |> EvalUpdateBundle.exec varEnv 
            vals.Get expr0, vals.Get expr1

    static member make (expr0: Expr<'T0>, expr1: Expr<'T1>, expr2: Expr<'T2>) =
        let bndl = 
            EvalUpdateBundle.empty
            |> EvalUpdateBundle.addExpr expr0
            |> EvalUpdateBundle.addExpr expr1
            |> EvalUpdateBundle.addExpr expr2
        fun (varEnv: VarEnv) ->
            let vals = bndl |> EvalUpdateBundle.exec varEnv 
            vals.Get expr0, vals.Get expr1, vals.Get expr2

    static member make (expr0: Expr<'T0>, expr1: Expr<'T1>, expr2: Expr<'T2>, expr3: Expr<'T3>) =
        let bndl = 
            EvalUpdateBundle.empty
            |> EvalUpdateBundle.addExpr expr0
            |> EvalUpdateBundle.addExpr expr1
            |> EvalUpdateBundle.addExpr expr2
            |> EvalUpdateBundle.addExpr expr3
        fun (varEnv: VarEnv) ->
            let vals = bndl |> EvalUpdateBundle.exec varEnv 
            vals.Get expr0, vals.Get expr1, vals.Get expr2, vals.Get expr3


    static member arg1 (vs0: Var<'T0>) (f: VarEnv -> 'TR) : Tensor<'T0> -> 'TR =
        fun (val0: Tensor<'T0>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> f

    static member arg2 (vs0: Var<'T0>) (vs1: Var<'T1>) (f: VarEnv -> 'TR) : Tensor<'T0> -> Tensor<'T1> -> 'TR =
        fun (val0: Tensor<'T0>) (val1: Tensor<'T1>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> f

    static member arg3 (vs0: Var<'T0>) (vs1: Var<'T1>) (vs2: Var<'T2>) f : Tensor<'T0> -> Tensor<'T1> -> Tensor<'T2> -> 'TR =
        fun (val0: Tensor<'T0>) (val1: Tensor<'T1>) (val2: Tensor<'T2>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> f   


    static member add (pi: ParSetInst) f =
        fun (ve: VarEnv) ->
            f (VarEnv.join ve pi.VarEnv)

