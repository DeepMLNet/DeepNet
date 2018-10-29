namespace SymTensor

open System

open Tensor
open Tensor.Utils
open Tensor.Backend
open ShapeSpec
open VarSpec
open ElemExpr


module ElemExprHostEval =


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
    let evalElement (expr: ElemExprT) (args: Tensor<'T> list) (idxs: ShapeSpecT) : 'T =
        let retType = (ElemExpr.typeName expr).Type
        if retType <> typeof<'T> then
            failwithf "elements expression of type %A does not match eval function of type %A"
                retType typeof<'T>                    

        let rec doEval symVals expr = 
            match expr with
            | Leaf (op) ->
                match op with
                | Const v -> v.GetConvertedValue()
                | SizeValue (ss, _) ->
                    let sv = ss |> SizeSpec.substSymbols symVals |> SizeSpec.eval
                    conv<'T> sv
                | ArgElement ((Arg n, argIdxs), _) ->
                    let argIdxs = ShapeSpec.substSymbols symVals argIdxs
                    let argIdxsVal = ShapeSpec.eval argIdxs
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
                    let first, last = SizeSpec.eval first, SizeSpec.eval last
                    (conv<'T> 0, [first .. last])
                    ||> List.fold (fun sumSoFar symVal ->
                        let sumElem = 
                            doEval (symVals |> Map.add sym (SizeSpec.fix symVal)) a
                        typedApply2 (unsp) (+) (+) (+) (+) sumSoFar sumElem) 
                | KroneckerRng (s, first, last) ->
                    let sVal = s |> SizeSpec.substSymbols symVals |> SizeSpec.eval
                    let firstVal = first |> SizeSpec.substSymbols symVals |> SizeSpec.eval
                    let lastVal = last |> SizeSpec.substSymbols symVals |> SizeSpec.eval
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
                    let leftVal = left |> SizeSpec.substSymbols symVals |> SizeSpec.eval
                    let rightVal = right |> SizeSpec.substSymbols symVals |> SizeSpec.eval
                    if leftVal = rightVal then av () else bv ()

        let initialSymVals =
            seq {for dim, idx in List.indexed idxs do yield (idxSymbol dim, idx)}
            |> Map.ofSeq
        doEval initialSymVals expr


    /// evaluates all elements of an element expression
    let eval (expr: ElemExprT) (args: Tensor<'T> list) (resShape: NShapeSpecT) =
        let res = HostTensor.zeros<'T> resShape
        for idx in TensorLayout.allIdxOfShape resShape do
            let symIdx = idx |> List.map SizeSpec.fix
            let ev = evalElement expr args symIdx 
            res.[idx] <- ev
        res
