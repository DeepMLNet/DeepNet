namespace SymTensor

open Basics
open ArrayNDNS
open ShapeSpec
open VarSpec
open System

module ElemExpr =

    /// argument 
    type ArgT =
    | Arg of int

    /// element of an argument
    type ArgElementSpecT = ArgT * ShapeSpecT
    

    type LeafOpT<'T> =
        | Const of 'T
        | SizeValue of SizeSpecT
        | ArgElement of ArgElementSpecT

    and UnaryOpT<'T> = 
        | Negate                        
        | Abs
        | SignT
        | Log
        | Log10                           
        | Exp                           
        | Sin
        | Cos
        | Tan
        | Asin
        | Acos
        | Atan
        | Sinh
        | Cosh
        | Tanh
        | Sqrt
        | Ceil
        | Floor
        | Round
        | Truncate
        | Sum of SizeSymbolT * SizeSpecT * SizeSpecT
        | KroneckerRng of SizeSpecT * SizeSpecT * SizeSpecT

    and BinaryOpT<'T> = 
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Modulo
        | Power        
        | IfThenElse of SizeSpecT * SizeSpecT
        
    /// an element expression
    and [<StructuredFormatDisplay("{PrettyString}")>]
        ElemExprT<'T> =
        | Leaf of LeafOpT<'T>
        | Unary of UnaryOpT<'T> * ElemExprT<'T>
        | Binary of BinaryOpT<'T> * ElemExprT<'T> * ElemExprT<'T>
        
    /// a constant value
    let scalar a =
        Leaf (Const a)
              
    type ElemExprT<'T> with

        // elementwise unary
        static member (~+) (a: ElemExprT<'T>) = a 
        static member (~-) (a: ElemExprT<'T>) = Unary(Negate, a)  
        static member Abs (a: ElemExprT<'T>) = Unary(Abs, a) 
        static member SignT (a: ElemExprT<'T>) = Unary(SignT, a) 
        static member Log (a: ElemExprT<'T>) = Unary(Log, a) 
        static member Log10 (a: ElemExprT<'T>) = Unary(Log10, a) 
        static member Exp (a: ElemExprT<'T>) = Unary(Exp, a) 
        static member Sin (a: ElemExprT<'T>) = Unary(Sin, a) 
        static member Cos (a: ElemExprT<'T>) = Unary(Cos, a) 
        static member Tan (a: ElemExprT<'T>) = Unary(Tan, a) 
        static member Asin (a: ElemExprT<'T>) = Unary(Asin, a) 
        static member Acos (a: ElemExprT<'T>) = Unary(Acos, a) 
        static member Atan (a: ElemExprT<'T>) = Unary(Atan, a) 
        static member Sinh (a: ElemExprT<'T>) = Unary(Sinh, a) 
        static member Cosh (a: ElemExprT<'T>) = Unary(Cosh, a) 
        static member Tanh (a: ElemExprT<'T>) = Unary(Tanh, a) 
        static member Sqrt (a: ElemExprT<'T>) = Unary(Sqrt, a) 
        static member Ceiling (a: ElemExprT<'T>) = Unary(Ceil, a) 
        static member Floor (a: ElemExprT<'T>) = Unary(Floor, a) 
        static member Round (a: ElemExprT<'T>) = Unary(Round, a) 
        static member Truncate (a: ElemExprT<'T>) = Unary(Truncate, a) 

        // elementwise binary
        static member (+) (a: ElemExprT<'T>, b: ElemExprT<'T>) = Binary(Add, a, b)
        static member (-) (a: ElemExprT<'T>, b: ElemExprT<'T>) = Binary(Substract, a, b)
        static member (*) (a: ElemExprT<'T>, b: ElemExprT<'T>) = Binary(Multiply, a, b)
        static member (/) (a: ElemExprT<'T>, b: ElemExprT<'T>) = Binary(Divide, a, b)
        static member (%) (a: ElemExprT<'T>, b: ElemExprT<'T>) = Binary(Modulo, a, b)
        static member Pow (a: ElemExprT<'T>, b: ElemExprT<'T>) = Binary(Power, a, b)

        // elementwise binary with basetype
        static member (+) (a: ElemExprT<'T>, b: 'T) = a + (scalar b)
        static member (-) (a: ElemExprT<'T>, b: 'T) = a - (scalar b)
        static member (*) (a: ElemExprT<'T>, b: 'T) = a * (scalar b)
        static member (/) (a: ElemExprT<'T>, b: 'T) = a / (scalar b)
        static member (%) (a: ElemExprT<'T>, b: 'T) = a % (scalar b)
        static member Pow (a: ElemExprT<'T>, b: 'T) = a ** (scalar b)

        static member (+) (a: 'T, b: ElemExprT<'T>) = (scalar a) + b
        static member (-) (a: 'T, b: ElemExprT<'T>) = (scalar a) - b
        static member (*) (a: 'T, b: ElemExprT<'T>) = (scalar a) * b
        static member (/) (a: 'T, b: ElemExprT<'T>) = (scalar a) / b
        static member (%) (a: 'T, b: ElemExprT<'T>) = (scalar a) % b
        static member Pow (a: 'T, b: ElemExprT<'T>) = (scalar a) ** b          
          
    /// sign keeping type
    let signt (a: ElemExprT<'T>) =
        ElemExprT<'T>.SignT a 

    /// square root
    let sqrtt (a: ElemExprT<'T>) =
        ElemExprT<'T>.Sqrt a       
        
    /// scalar of given value and type
    let inline scalart<'T> f = scalar (conv<'T> f)

    /// scalar 0 of appropriate type
    let inline zero<'T> () = scalar (ArrayNDT<'T>.Zero)

    /// scalar 1 of appropriate type
    let inline one<'T> () = scalar (ArrayNDT<'T>.One)

    /// scalar 2 of appropriate type
    let inline two<'T> () = scalart<'T> 2
           
    /// index symbol for given dimension of the result
    let idxSymbol dim =
        sprintf "R%d" dim
        |> SizeSymbol.ofName

    /// index size of given dimension of the result
    let idx dim =
        Base (Sym (idxSymbol dim))

    /// summation symbol of given name
    let sumSymbol name =
        sprintf "SUM_%s" name 
        |> SizeSymbol.ofName

    /// summation index size of given name
    let sumIdx name =
        Base (Sym (sumSymbol name))

    /// sum exprover given index (created by sumIdx) from first to last
    let sum idx first last expr =
        match idx with
        | Base (Sym (sumSym)) ->
            Unary (Sum (sumSym, first, last), expr) 
        | _ -> invalidArg "idx" "idx must be summation index obtained by calling sumIdx"

    /// If left=right, then thenExpr else elseExpr.
    let ifThenElse left right thenExpr elseExpr =
        Binary (IfThenElse (left, right), thenExpr, elseExpr)

    /// expr if first <= sym <= last, otherwise 0.
    let kroneckerRng sym first last expr =
        Unary (KroneckerRng (sym, first, last), expr)

    /// the element with index idx of the n-th argument
    let argElem pos idx =
        Leaf (ArgElement (Arg pos, idx))

    type Argument1D (pos: int) =       
        member this.Item with get (i0) = argElem pos [i0]
    type Argument2D (pos: int) =       
        member this.Item with get (i0, i1) = argElem pos [i0; i1]
    type Argument3D (pos: int) =       
        member this.Item with get (i0, i1, i2) = argElem pos [i0; i1; i2]
    type Argument4D (pos: int) =       
        member this.Item with get (i0, i1, i2, i3) = argElem pos [i0; i1; i2; i3]
    type Argument5D (pos: int) =       
        member this.Item with get (i0, i1, i2, i3, i4) = argElem pos [i0; i1; i2; i3; i4]
    type Argument6D (pos: int) =       
        member this.Item with get (i0, i1, i2, i3, i4, i5) = argElem pos [i0; i1; i2; i3; i4; i5]

    /// 1-dimensional argument at given position
    let arg1D pos = Argument1D pos
    /// 2-dimensional argument at given position
    let arg2D pos = Argument2D pos
    /// 3-dimensional argument at given position
    let arg3D pos = Argument3D pos
    /// 4-dimensional argument at given position
    let arg4D pos = Argument4D pos
    /// 5-dimensional argument at given position
    let arg5D pos = Argument5D pos
    /// 6-dimensional argument at given position
    let arg6D pos = Argument6D pos
   
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
    let evalElement (expr: ElemExprT<'T>) (args: ArrayNDT<'T> list) (idxs: ShapeSpecT) =

        let rec doEval symVals expr = 
            match expr with
            | Leaf (op) ->
                match op with
                | Const v -> v
                | SizeValue ss ->
                    let sv = ss |> SizeSpec.substSymbols symVals |> SizeSpec.eval
                    conv<'T> sv
                | ArgElement (Arg n, argIdxs) ->
                    let argIdxs = ShapeSpec.substSymbols symVals argIdxs
                    let argIdxsVal = ShapeSpec.eval argIdxs
                    args.[n] |> ArrayND.get argIdxsVal

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
    let eval (expr: ElemExprT<'T>) (args: ArrayNDT<'T> list) (resShape: NShapeSpecT) =
        let res = ArrayNDHost.zeros<'T> resShape
        for idx in ArrayNDLayout.allIdxOfShape resShape do
            let symIdx = idx |> List.map SizeSpec.fix
            let ev = evalElement expr args symIdx 
            ArrayND.set idx ev res
        res

    /// returns the required number of arguments of the element expression
    let rec requiredNumberOfArgs expr =
        match expr with
        | Leaf (ArgElement (Arg n, _)) -> 
            if n < 0 then failwith "argument index must be positive"
            n
        | Leaf _ -> 0

        | Unary (op, a) -> requiredNumberOfArgs a
        | Binary (op, a, b) -> max (requiredNumberOfArgs a) (requiredNumberOfArgs b)
    
    /// checks if the arguments' shapes are compatible with the result shape        
    let checkArgShapes (expr: ElemExprT<'T>) (argShapes: ShapeSpecT list) (resShape: ShapeSpecT) =
        // check number of arguments
        let nArgs = List.length argShapes
        let nReqArgs = requiredNumberOfArgs expr       
        if nReqArgs > nArgs then
            failwithf "the element expression requires at least %d arguments but only %d arguments were specified"
                nReqArgs nArgs

        // check dimensionality of arguments
        let rec checkDims expr =
            match expr with
            | Leaf (ArgElement (Arg n, idx)) ->
                let idxDim = ShapeSpec.nDim idx
                let argDim = ShapeSpec.nDim argShapes.[n]
                if idxDim <> argDim then
                    failwithf 
                        "the argument with zero-based index %d has %d dimensions but was used  \
                         with %d dimensions in the element expression" n argDim idxDim
            | Leaf _ -> ()

            | Unary (_, a) -> checkDims a
            | Binary (_, a, b) -> checkDims a; checkDims b
        checkDims expr

    /// substitutes the specified size symbols with their replacements 
    let rec substSymSizes symSizes expr = 
        let sSub = substSymSizes symSizes
        let sSize = SymSizeEnv.subst symSizes
        let sShp = SymSizeEnv.substShape symSizes

        match expr with
        | Leaf (SizeValue sc) -> Leaf (SizeValue (sSize sc))
        | Leaf (ArgElement (arg, argIdxs)) -> Leaf (ArgElement (arg, sShp argIdxs))
        | Leaf _ -> expr

        | Unary (Sum (sym, first, last), a) -> 
            Unary (Sum (sym, sSize first, sSize last), sSub a)
        | Unary (KroneckerRng (sym, first, last), a) ->
            Unary (KroneckerRng (sSize sym, sSize first, sSize last), sSub a)
        | Unary (op, a) -> Unary (op, sSub a)
        | Binary (IfThenElse (left, right), a, b) ->
            Binary (IfThenElse (sSize left, sSize right), sSub a, sSub b)
        | Binary (op, a, b) -> Binary (op, sSub a, sSub b)

    /// true if all size symbols can be evaluated to numeric values 
    let canEvalAllSymSizes expr =
        let rec canEval expr =  
            match expr with
            | Leaf (SizeValue sc) -> SizeSpec.canEval sc
            | Leaf (ArgElement (arg, argIdxs)) -> ShapeSpec.canEval argIdxs
            | Leaf _ -> true

            | Unary (Sum (sym, first, last), a) -> 
                // replace sum index by one before testing for evaluability
                let sumSymVals = Map [sym, SizeSpec.one]
                SizeSpec.canEval first && 
                SizeSpec.canEval last && 
                canEval (a |> substSymSizes sumSymVals)
            | Unary (KroneckerRng (sym, first, last), a) ->
                SizeSpec.canEval sym && 
                SizeSpec.canEval first && 
                SizeSpec.canEval last &&
                canEval a
            | Unary (op, a) -> canEval a
            | Binary (IfThenElse (left, right), a, b) ->
                SizeSpec.canEval left && 
                SizeSpec.canEval right &&
                canEval a && canEval b
            | Binary (op, a, b) -> canEval a && canEval b

        // replace output indices by ones before testing for evaluability
        let dummySymVals =
            seq {for dim=0 to 20 do yield (idxSymbol dim, SizeSpec.one)}
            |> Map.ofSeq
        expr |> substSymSizes dummySymVals |> canEval

    /// pretty string of an element expression
    let rec prettyString (expr: ElemExprT<'T>) =
        // TODO: delete unnecessary brackets
        match expr with
        | Leaf (op) -> 
            match op with
            | Const v -> sprintf "%A" v
            | SizeValue ss -> sprintf "%A" ss
            | ArgElement ((Arg a), idxs) -> sprintf "a%d%A" a idxs
        
        | Unary (op, a) ->
            match op with
            | Negate -> sprintf "-(%s)" (prettyString a)
            | Abs -> sprintf "abs(%s)" (prettyString a)
            | SignT -> sprintf "signt(%s)" (prettyString a) 
            | Log -> sprintf "log(%s)" (prettyString a)
            | Log10 -> sprintf "log10(%s)" (prettyString a)
            | Exp -> sprintf "exp(%s)" (prettyString a)
            | Sin -> sprintf "sin(%s)" (prettyString a)
            | Cos -> sprintf "cos(%s)" (prettyString a)
            | Tan -> sprintf "tan(%s)" (prettyString a)
            | Asin -> sprintf "asin(%s)" (prettyString a)
            | Acos -> sprintf "acos(%s)" (prettyString a)
            | Atan -> sprintf "atan(%s)" (prettyString a)
            | Sinh -> sprintf "sinh(%s)" (prettyString a)
            | Cosh -> sprintf "cosh(%s)" (prettyString a)
            | Tanh -> sprintf "tanh(%s)" (prettyString a)
            | Sqrt -> sprintf "sqrt(%s)" (prettyString a)
            | Ceil -> sprintf "ceil(%s)" (prettyString a) 
            | Floor -> sprintf "floor(%s)" (prettyString a) 
            | Round -> sprintf "round(%s)" (prettyString a)
            | Truncate -> sprintf "truncate(%s)" (prettyString a)
            | Sum (sumSym, first, last)-> 
                sprintf "sum(%A[%A..%A], %s)" sumSym first last (prettyString a)
            | KroneckerRng (s, first, last) ->
                sprintf "%s[%s%s%s %s %s%s%s](%s)"  (String('\u03B4',1)) 
                                                    (prettyString (Leaf (SizeValue first)))
                                                    (String('\u2264',1))
                                                    (prettyString (Leaf (SizeValue s)))
                                                    (String('\u2227',1))
                                                    (prettyString (Leaf (SizeValue s)))
                                                    (String('\u2264',1))
                                                    (prettyString (Leaf (SizeValue last)))
                                                    (prettyString a)
        | Binary(op, a, b) -> 
            match op with
            | Add -> sprintf "(%s + %s)" (prettyString a) (prettyString b)
            | Substract -> sprintf "(%s - %s)" (prettyString a) (prettyString b)
            | Multiply -> sprintf "(%s * %s)" (prettyString a) (prettyString b)
            | Divide -> sprintf "(%s / %s)" (prettyString a) (prettyString b)
            | Modulo -> sprintf "(%s %% %s)" (prettyString a) (prettyString b)
            | Power -> sprintf "(%s ** %s)" (prettyString a) (prettyString b)
            | IfThenElse (left,right)-> 
                sprintf "ifThenElse(%A = %A, %s, %s)" left right (prettyString a) (prettyString b)

    type ElemExprT<'T> with
        member this.PrettyString = prettyString this

[<AutoOpen>]
module ElemExprTypes =
    type ElemExprT<'T> = ElemExpr.ElemExprT<'T>
