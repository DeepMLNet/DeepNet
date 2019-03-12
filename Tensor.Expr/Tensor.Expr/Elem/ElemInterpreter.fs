namespace Tensor.Expr.Elem

open Tensor.Expr
open DeepNet.Utils


/// Slow and simple interpreter for evaluating element expressions.
module Interpreter =

    let inline uncheckedApply (f: 'T -> 'T) (a: 'S) : 'S =
        let av = a |> box |> unbox
        f av |> box |> unbox

    let inline uncheckedApply2 (f: 'T -> 'T -> 'T) (a: 'S) (b: 'S) : 'S =
        let av = a |> box |> unbox
        let bv = b |> box |> unbox
        f av bv |> box |> unbox

    let inline typedApply   (fBool:   bool   -> bool) 
                            (fDouble: double -> double) 
                            (fSingle: single -> single)
                            (fInt:    int    -> int)
                            (fByte:   byte   -> byte)
                            (a: 'T) : 'T =
        if   typeof<'T>.Equals(typeof<bool>)   then uncheckedApply fBool a
        elif typeof<'T>.Equals(typeof<double>) then uncheckedApply fDouble a 
        elif typeof<'T>.Equals(typeof<single>) then uncheckedApply fSingle a 
        elif typeof<'T>.Equals(typeof<int>)    then uncheckedApply fInt    a 
        elif typeof<'T>.Equals(typeof<byte>)   then uncheckedApply fByte   a 
        else failwith "unknown type"

    let inline typedApply2   (fBool:   bool   -> bool   -> bool) 
                                (fDouble: double -> double -> double) 
                                (fSingle: single -> single -> single)
                                (fInt:    int    -> int    -> int)
                                (fByte:   byte   -> byte   -> byte)
                                (a: 'T) (b: 'T) : 'T =
        if   typeof<'T>.Equals(typeof<bool>)   then uncheckedApply2 fBool   a b
        elif typeof<'T>.Equals(typeof<double>) then uncheckedApply2 fDouble a b 
        elif typeof<'T>.Equals(typeof<single>) then uncheckedApply2 fSingle a b
        elif typeof<'T>.Equals(typeof<int>)    then uncheckedApply2 fInt    a b
        elif typeof<'T>.Equals(typeof<byte>)   then uncheckedApply2 fByte   a b
        else failwith "unknown type"

    /// unsupported operation for this type
    let inline unsp (a: 'T) : 'R = 
        failwithf "operation unsupported for type %A" typeof<'T>

    let inline signImpl (x: 'T) =
        conv<'T> (sign x)

    /// evaluates the specified element of an element expression
    let evalElement (expr: Expr) (args: Tensor.Tensor<'T> list) (idxs: Shape) : 'T =
        if expr.Type <> typeof<'T> then
            failwithf "Elem.Expr of type %A does not match eval function of type %A."
                expr.Type typeof<'T>                    

        let rec doEval symVals expr = 
            match expr with
            | Leaf (op) ->
                match op with
                | LeafOp.Const v -> unbox v.Value
                | SizeValue (ss, _) ->
                    let sv = ss |> Size.substSyms symVals |> Size.eval
                    conv<'T> sv
                | ArgElement ((Arg n, argIdxs), _) ->
                    let argIdxs = Shape.substSymbols symVals argIdxs
                    let argIdxsVal = Shape.eval argIdxs
                    args.[n].[argIdxsVal]

            | Unary (op, a) ->
                let av () = doEval symVals a
                match op with
                | Negate ->   typedApply (unsp) (~-) (~-) (~-) (unsp) (av ())
                | Abs ->      typedApply (unsp) abs abs abs (unsp) (av ())
                | SignT ->    typedApply (unsp) signImpl signImpl sign (unsp) (av ())
                | Log ->      typedApply (unsp) log log (unsp) (unsp) (av ())
                | Log10 ->    typedApply (unsp) log10 log10 (unsp) (unsp) (av ())
                | Exp ->      typedApply (unsp) exp exp (unsp) (unsp) (av ())
                | Sin ->      typedApply (unsp) sin sin (unsp) (unsp) (av ())
                | Cos ->      typedApply (unsp) cos cos (unsp) (unsp) (av ())
                | Tan ->      typedApply (unsp) tan tan (unsp) (unsp) (av ())
                | Asin ->     typedApply (unsp) asin asin (unsp) (unsp) (av ())
                | Acos ->     typedApply (unsp) acos acos (unsp) (unsp) (av ())
                | Atan ->     typedApply (unsp) atan atan (unsp) (unsp) (av ())
                | Sinh ->     typedApply (unsp) sinh sinh (unsp) (unsp) (av ())
                | Cosh ->     typedApply (unsp) cosh cosh (unsp) (unsp) (av ())
                | Tanh ->     typedApply (unsp) tanh tanh (unsp) (unsp) (av ())
                | Sqrt ->     typedApply (unsp) sqrt sqrt (unsp) (unsp) (av ())
                | Ceil ->     typedApply (unsp) ceil ceil (unsp) (unsp) (av ())
                | Floor ->    typedApply (unsp) floor floor (unsp) (unsp) (av ())
                | Round ->    typedApply (unsp) round round (unsp) (unsp) (av ())
                | Truncate -> typedApply (unsp) truncate truncate (unsp) (unsp) (av ())
                | Sum (sym, first, last) ->
                    let first, last = Size.eval first, Size.eval last
                    (conv<'T> 0, [first .. last])
                    ||> List.fold (fun sumSoFar symVal ->
                        let sumElem = 
                            doEval (symVals |> Map.add sym (Size.fix symVal)) a
                        typedApply2 (unsp) (+) (+) (+) (+) sumSoFar sumElem) 
                | KroneckerRng (s, first, last) ->
                    let sVal = s |> Size.substSyms symVals |> Size.eval
                    let firstVal = first |> Size.substSyms symVals |> Size.eval
                    let lastVal = last |> Size.substSyms symVals |> Size.eval
                    if firstVal <= sVal && sVal <= lastVal then av ()
                    else conv<'T> 0

            | Binary (op, a, b) ->
                let av () = doEval symVals a
                let bv () = doEval symVals b
                match op with
                | Add  ->       typedApply2 (unsp) (+) (+) (+) (+) (av()) (bv())                 
                | Substract ->  typedApply2 (unsp) (-) (-) (-) (-) (av()) (bv())              
                | Multiply ->   typedApply2 (unsp) (*) (*) (*) (*) (av()) (bv())                  
                | Divide ->     typedApply2 (unsp) (/) (/) (/) (/) (av()) (bv())                    
                | Modulo ->     typedApply2 (unsp) (%) (%) (%) (%) (av()) (bv())
                | Power ->      typedApply2 (unsp) ( ** ) ( ** ) (unsp) (unsp) (av()) (bv())
                | IfThenElse (left, right) ->
                    let leftVal = left |> Size.substSyms symVals |> Size.eval
                    let rightVal = right |> Size.substSyms symVals |> Size.eval
                    if leftVal = rightVal then av () else bv ()

        let initialSymVals =
            seq {for dim, idx in List.indexed idxs do yield (Expr.idxSymbol dim, idx)}
            |> Map.ofSeq
        doEval initialSymVals expr

    /// Evaluates all elements of an element expression.
    let eval (expr: Expr) (args: Tensor.Tensor<'T> list) (resShape: int64 list) : Tensor.Tensor<'T> =
        let res = Tensor.HostTensor.zeros<'T> resShape
        for idx in Tensor.Backend.TensorLayout.allIdxOfShape resShape do
            let symIdx = idx |> List.map Size.fix
            let ev = evalElement expr args symIdx 
            res.[idx] <- ev
        res


type internal IInterpreter = 
    abstract Eval: Expr -> Tensor.ITensor list -> int64 list -> Tensor.ITensor
type internal TInterpreter<'T> () =
    interface IInterpreter with 
        member this.Eval expr args resShape =
            let args = args |> List.map (fun a -> a :?> Tensor.Tensor<'T>)
            Interpreter.eval expr args resShape :> Tensor.ITensor

type Interpreter =
    /// Evaluates all elements of an element expression using untyped tensors.
    static member evalUntyped (expr: Expr) (args: Tensor.ITensor list) (resShape: int64 list) : Tensor.ITensor = 
        (Generic<TInterpreter<_>, IInterpreter> [args.Head.DataType]).Eval expr args resShape
    
