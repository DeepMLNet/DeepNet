namespace Tensor


/// Useful core operators.
[<AutoOpen>]
module Operators =

    type internal SgnDynamicImplTable<'T>() = 
        static let result : ('T -> 'T) = 
            let aty = typeof<'T>
            if   aty.Equals(typeof<int16>)    then unbox(fun (x:int16)  -> if x < 0s then -1s elif x > 0s then 1s else 0s)
            elif aty.Equals(typeof<int32>)    then unbox(fun (x:int32)  -> if x < 0  then -1  elif x > 0  then 1  else 0)
            elif aty.Equals(typeof<int64>)    then unbox(fun (x:int64)  -> if x < 0L then -1L elif x > 0L then 1L else 0L)
            elif aty.Equals(typeof<single>)   then unbox(fun (x:single) -> if x < 0.0f then -1.0f elif x > 0.0f then 1.0f else 0.0f)
            elif aty.Equals(typeof<double>)   then unbox(fun (x:double) -> if x < 0.0  then -1.0  elif x > 0.0  then 1.0  else 0.0)
            else
                 let mi = aty.GetMethod("Sgn", [|aty|])
                 (fun x -> unbox(mi.Invoke(null, [|box x|])))
        static member Result : ('T -> 'T) = result

    /// Sign of value returned using same type.
    /// In contrast, the F# builtin function sign returns an int.
    /// Calls static method Sgn on non-primitive types.
    [<CompiledName("Sgn")>]
    let sgn x = 
        SgnDynamicImplTable<_>.Result x 
        



