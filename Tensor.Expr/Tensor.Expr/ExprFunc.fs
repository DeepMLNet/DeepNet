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

    static member make (expr0: Expr<'T0>, expr1: Expr<'T1>, expr2: Expr<'T2>, expr3: Expr<'T3>, expr4: Expr<'T4>) =
        let bndl = 
            EvalUpdateBundle.empty
            |> EvalUpdateBundle.addExpr expr0
            |> EvalUpdateBundle.addExpr expr1
            |> EvalUpdateBundle.addExpr expr2
            |> EvalUpdateBundle.addExpr expr3
            |> EvalUpdateBundle.addExpr expr4
        fun (varEnv: VarEnv) ->
            let vals = bndl |> EvalUpdateBundle.exec varEnv 
            vals.Get expr0, vals.Get expr1, vals.Get expr2, vals.Get expr3, vals.Get expr4


    static member arg1 (vs0: Var<'T0>) (f: VarEnv -> 'TR) : Tensor<'T0> -> 'TR =
        fun (val0: Tensor<'T0>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> f

    static member arg2 (vs0: Var<'T0>) (vs1: Var<'T1>) (f: VarEnv -> 'TR) : Tensor<'T0> -> Tensor<'T1> -> 'TR =
        fun (val0: Tensor<'T0>) (val1: Tensor<'T1>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> f

    static member arg3 (vs0: Var<'T0>) (vs1: Var<'T1>) (vs2: Var<'T2>) f : Tensor<'T0> -> Tensor<'T1> -> Tensor<'T2> -> 'TR =
        fun (val0: Tensor<'T0>) (val1: Tensor<'T1>) (val2: Tensor<'T2>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> f   

    static member arg4 (vs0: Var<'T0>) (vs1: Var<'T1>) (vs2: Var<'T2>) (vs3: Var<'T3>) f : Tensor<'T0> -> Tensor<'T1> -> Tensor<'T2> -> Tensor<'T3> -> 'TR =
        fun (val0: Tensor<'T0>) (val1: Tensor<'T1>) (val2: Tensor<'T2>) (val3: Tensor<'T3>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> VarEnv.add vs3 val3 |> f   

    static member arg5 (vs0: Var<'T0>) (vs1: Var<'T1>) (vs2: Var<'T2>) (vs3: Var<'T3>) (vs4: Var<'T4>) f : Tensor<'T0> -> Tensor<'T1> -> Tensor<'T2> -> Tensor<'T3> -> Tensor<'T4> -> 'TR =
        fun (val0: Tensor<'T0>) (val1: Tensor<'T1>) (val2: Tensor<'T2>) (val3: Tensor<'T3>) (val4: Tensor<'T4>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> VarEnv.add vs3 val3 |> VarEnv.add vs4 val4 |> f   


    static member add (pi: ParSetInst) f =
        fun (ve: VarEnv) ->
            f (VarEnv.join ve pi.VarEnv)

