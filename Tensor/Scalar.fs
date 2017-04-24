namespace ArrayNDNS

open System
open System.Linq.Expressions
open System.Collections.Generic

open Basics

type ScalarOps<'T>() =

    static let instances = Dictionary<Type, obj>()

    let a = Expression.Parameter(typeof<'T>, "a")
    let b = Expression.Parameter(typeof<'T>, "b")

    member val PlusFunc = 
        Expression.Lambda<Func<'T, 'T, 'T>>(Expression.Add(a, b), a, b).Compile()
    member inline this.Plus a b = this.PlusFunc.Invoke(a, b)
        



    static member Get<'T> () =
        match instances.TryFind typeof<'T> with
        | Some inst -> inst :?> ScalarOps<'T>
        | None ->
            let inst = ScalarOps<'T> ()
            instances.Add (typeof<'T>, inst)
            inst
        
