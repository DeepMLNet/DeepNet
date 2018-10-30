namespace Tensor.Utils

open System
open System.Reflection
open System.Linq.Expressions

open DeepNet.Utils


/// Generic scalar operation primitives.
type internal ScalarPrimitives<'T, 'TC> () =

    static let fscAsm = Assembly.GetAssembly(typeof<unit>)
    static let myAsm = Assembly.GetExecutingAssembly()
    static let fso = fscAsm.GetType("Microsoft.FSharp.Core.Operators", true)
    static let tso = myAsm.GetType("Tensor.Operators", true)

    static let a = Expression.Parameter(typeof<'T>, "a")
    static let b = Expression.Parameter(typeof<'T>, "b")
    static let c = Expression.Parameter(typeof<'TC>, "c")
    static let cond = Expression.Parameter(typeof<bool>, "cond")

    static let compileAny (fns: (unit -> Expression<_>) list) =        
        match fns |> List.tryPick (fun fn ->
                try Some (fn().Compile())
                with :? InvalidOperationException -> None) with
        | Some expr -> expr
        | None -> 
            failwithf "cannot compile scalar primitive for type %s" typeof<'T>.Name

    static let tryUnary op fns =
        let errExpr () =
            let msg = sprintf "the type %s does not implemented %s" typeof<'T>.Name op
            let thrw = Expression.Throw(Expression.Constant(InvalidOperationException msg))
            Expression.Lambda<Func<'T, 'T>>(Expression.Block(thrw, a), a)
        compileAny (fns @ [errExpr])

    static let tryBinary op fns =
        let errExpr () =
            let msg = sprintf "the type %s does not implemented %s" typeof<'T>.Name op
            let thrw = Expression.Throw(Expression.Constant(InvalidOperationException msg))
            Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Block(thrw, a), a, b)
        compileAny (fns @ [errExpr])

    static let tryCompare op fns =
        let errExpr () =
            let msg = sprintf "the type %s does not implemented %s" typeof<'T>.Name op
            let thrw = Expression.Throw(Expression.Constant(InvalidOperationException msg))
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.Block(thrw, Expression.Constant(false)), a, b)
        compileAny (fns @ [errExpr])

    member val ConvertFunc = 
        Expression.Lambda<Func<'TC, 'T>>(Expression.Convert(c, typeof<'T>), c).Compile()
    member inline this.Convert cv = this.ConvertFunc.Invoke(cv)

    member val UnaryPlusFunc = 
        tryUnary "~+" [fun () -> Expression.Lambda<Func<'T, 'T>>(Expression.UnaryPlus(a), a)]
    member inline this.UnaryPlus av = this.UnaryPlusFunc.Invoke(av)

    member val UnaryMinusFunc = 
        tryUnary "~-" [fun () -> Expression.Lambda<Func<'T, 'T>>(Expression.Negate(a), a)]
    member inline this.UnaryMinus av = this.UnaryMinusFunc.Invoke(av)

    member val AbsFunc = 
        let m = fso.GetMethod("Abs").MakeGenericMethod (typeof<'T>)   
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()   
    member inline this.Abs av = this.AbsFunc.Invoke(av)

    member val SgnFunc = 
        let m = tso.GetMethod("Sgn").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Sgn av = this.SgnFunc.Invoke(av)

    member val LogFunc = 
        let m = fso.GetMethod("Log").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Log av = this.LogFunc.Invoke(av)

    member val Log10Func = 
        let m = fso.GetMethod("Log10").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Log10 av = this.Log10Func.Invoke(av)

    member val ExpFunc = 
        let m = fso.GetMethod("Exp").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Exp av = this.ExpFunc.Invoke(av)

    member val SinFunc = 
        let m = fso.GetMethod("Sin").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Sin av = this.SinFunc.Invoke(av)

    member val CosFunc = 
        let m = fso.GetMethod("Cos").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Cos av = this.CosFunc.Invoke(av)

    member val TanFunc = 
        let m = fso.GetMethod("Tan").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Tan av = this.TanFunc.Invoke(av)

    member val AsinFunc = 
        let m = fso.GetMethod("Asin").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Asin av = this.AsinFunc.Invoke(av)

    member val AcosFunc = 
        let m = fso.GetMethod("Acos").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Acos av = this.AcosFunc.Invoke(av)

    member val AtanFunc = 
        let m = fso.GetMethod("Atan").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Atan av = this.AtanFunc.Invoke(av)

    member val SinhFunc = 
        let m = fso.GetMethod("Sinh").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Sinh av = this.SinhFunc.Invoke(av)

    member val CoshFunc = 
        let m = fso.GetMethod("Cosh").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Cosh av = this.CoshFunc.Invoke(av)

    member val TanhFunc = 
        let m = fso.GetMethod("Tanh").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Tanh av = this.TanhFunc.Invoke(av)

    member val SqrtFunc = 
        let m = fso.GetMethod("Sqrt").MakeGenericMethod (typeof<'T>, typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Sqrt av = this.SqrtFunc.Invoke(av)

    member val CeilingFunc = 
        let m = fso.GetMethod("Ceiling").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Ceiling av = this.CeilingFunc.Invoke(av)

    member val FloorFunc = 
        let m = fso.GetMethod("Floor").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Floor av = this.FloorFunc.Invoke(av)

    member val RoundFunc = 
        let m = fso.GetMethod("Round").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Round av = this.RoundFunc.Invoke(av)

    member val TruncateFunc = 
        let m = fso.GetMethod("Truncate").MakeGenericMethod (typeof<'T>)        
        Expression.Lambda<Func<'T, 'T>>(Expression.Call(m, a), a).Compile()
    member inline this.Truncate av = this.TruncateFunc.Invoke(av)

    member val AddFunc = 
        tryBinary "+" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Add(a, b), a, b)]
    member inline this.Add av bv = this.AddFunc.Invoke(av, bv)
        
    member val SubtractFunc = 
        tryBinary "-" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Subtract(a, b), a, b)]
    member inline this.Subtract av bv = this.SubtractFunc.Invoke(av, bv)

    member val MultiplyFunc = 
        tryBinary "*" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Multiply(a, b), a, b)]
    member inline this.Multiply av bv = this.MultiplyFunc.Invoke(av, bv)

    member val DivideFunc = 
        tryBinary "/" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Divide(a, b), a, b)]
    member inline this.Divide av bv = this.DivideFunc.Invoke(av, bv)

    member val ModuloFunc = 
        tryBinary "%" [fun () -> Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Modulo(a, b), a, b)]
    member inline this.Modulo av bv = this.ModuloFunc.Invoke(av, bv)

    member val PowerFunc = 
        // note: power is currently significantly slower than other operations
        let m = fso.GetMethod("op_Exponentiation").MakeGenericMethod (typeof<'T>, typeof<'T>)        
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Call(m, a, b), a, b).Compile()
    member inline this.Power av bv = this.PowerFunc.Invoke(av, bv)

    member val IsFiniteFunc : ('T -> bool) =
        match typeof<'T> with
        | t when t=typeof<single> -> 
            unbox (fun (v: single) -> not (System.Single.IsInfinity v || System.Single.IsNaN v))
        | t when t=typeof<double> -> 
            unbox (fun (v: double) -> not (System.Double.IsInfinity v || System.Double.IsNaN v))
        | _ -> (fun _ -> true)
    member inline this.IsFinite av = this.IsFiniteFunc av

    member val EqualFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.Equal(a, b), a, b)
        let fnInterface () =
            let m = typeof<IEquatable<'T>>.GetMethod("Equals") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.Call(a, m, b), a, b)
        tryCompare "=" [fnOp; fnInterface]          
    member inline this.Equal av bv = this.EqualFunc.Invoke(av, bv)

    member val NotEqualFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.NotEqual(a, b), a, b)
        let fnInterface () =
            let m = typeof<IEquatable<'T>>.GetMethod("Equals") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.IsFalse(Expression.Call(a, m, b)), a, b)
        tryCompare "!=" [fnOp; fnInterface]      
    member inline this.NotEqual av bv = this.NotEqualFunc.Invoke(av, bv)

    member val LessFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThan(a, b), a, b)
        let fnInterface () =
            let m = typeof<IComparable<'T>>.GetMethod("CompareTo") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThan(Expression.Call(a, m, b), 
                                                                      Expression.Constant(0)), a, b)
        tryCompare "<" [fnOp; fnInterface]
    member inline this.Less av bv = this.LessFunc.Invoke(av, bv)

    member val LessOrEqualFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThanOrEqual(a, b), a, b)
        let fnInterface () =
            let m = typeof<IComparable<'T>>.GetMethod("CompareTo") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.LessThanOrEqual(Expression.Call(a, m, b), 
                                                                            Expression.Constant(0)), a, b)
        tryCompare "<=" [fnOp; fnInterface]
    member inline this.LessOrEqual av bv = this.LessOrEqualFunc.Invoke(av, bv)

    member val GreaterFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThan(a, b), a, b)
        let fnInterface () =
            let m = typeof<IComparable<'T>>.GetMethod("CompareTo") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThan(Expression.Call(a, m, b), 
                                                                         Expression.Constant(0)), a, b)
        tryCompare ">" [fnOp; fnInterface]
    member inline this.Greater av bv = this.GreaterFunc.Invoke(av, bv)

    member val GreaterOrEqualFunc = 
        let fnOp () = Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThanOrEqual(a, b), a, b)
        let fnInterface () =
            let m = typeof<IComparable<'T>>.GetMethod("CompareTo") 
            Expression.Lambda<Func<'T, 'T, bool>>(Expression.GreaterThanOrEqual(Expression.Call(a, m, b), 
                                                                                Expression.Constant(0)), a, b)
        tryCompare ">=" [fnOp; fnInterface]
    member inline this.GreaterOrEqual av bv = this.GreaterOrEqualFunc.Invoke(av, bv)

    member val IfThenElseFunc =
        Expression.Lambda<Func<bool, 'T, 'T, 'T>>(Expression.Condition(cond, a, b), cond, a, b).Compile()
    member inline this.IfThenElse condv ifTrue ifFalse = this.IfThenElseFunc.Invoke(condv, ifTrue, ifFalse)


/// Generic scalar operation primitives.
module internal ScalarPrimitives = 
    let private instances = Dictionary<Type * Type, obj>()
    let For<'T, 'TC> () =
        lock instances (fun () ->
            let types = typeof<'T>, typeof<'TC>
            match instances.TryFind types with
            | Some inst -> inst :?> ScalarPrimitives<'T, 'TC>
            | None ->
                let inst = ScalarPrimitives<'T, 'TC> ()
                instances.Add (types, inst)
                inst
        )
