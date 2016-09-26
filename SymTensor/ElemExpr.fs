namespace SymTensor

open Basics
open ArrayNDNS
open ShapeSpec
open VarSpec
open System

/// element expression
module ElemExpr =

    /// argument 
    type ArgT = Arg of pos:int

    /// element of an argument
    type ArgElementSpecT = ArgT * ShapeSpecT
    
    type LeafOpT =
        | Const of ConstSpecT
        | SizeValue of value:SizeSpecT * typ:TypeNameT
        | ArgElement of argElem:ArgElementSpecT * typ:TypeNameT

    and UnaryOpT = 
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

    and BinaryOpT = 
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Modulo
        | Power        
        | IfThenElse of SizeSpecT * SizeSpecT
        
    /// an element expression
    and [<StructuredFormatDisplay("{PrettyString}")>]
        ElemExprT =
        | Leaf of LeafOpT
        | Unary of UnaryOpT * ElemExprT
        | Binary of BinaryOpT * ElemExprT * ElemExprT
        
    /// data type name of the element expression
    let rec typeName expr =
        match expr with
        | Leaf (Const cs) -> cs.TypeName
        | Leaf (SizeValue (_, tn)) -> tn
        | Leaf (ArgElement (_, tn)) -> tn

        | Unary (_, a) -> typeName a
        | Binary (_, a, b) ->
            let ta, tb = typeName a, typeName b
            if ta <> tb then
                failwithf "ElemExpr must be of one type only but got types %A and %A"
                    ta.Type tb.Type
            ta

    type ElemExprT with
        /// typename of data
        member this.TypeName = typeName this

        /// type of data
        member this.Type = TypeName.getType this.TypeName

    /// checks the elements expression
    let check expr =
        typeName expr |> ignore
        expr

    /// a constant value
    let scalar f =
        Leaf (Const (ConstSpec.ofValue f)) |> check
              
    type ElemExprT with

        // elementwise unary
        static member (~+) (a: ElemExprT) = a |> check
        static member (~-) (a: ElemExprT) = Unary(Negate, a) |> check
        static member Abs (a: ElemExprT) = Unary(Abs, a) |> check
        static member SignT (a: ElemExprT) = Unary(SignT, a) |> check
        static member Log (a: ElemExprT) = Unary(Log, a) |> check
        static member Log10 (a: ElemExprT) = Unary(Log10, a) |> check
        static member Exp (a: ElemExprT) = Unary(Exp, a) |> check
        static member Sin (a: ElemExprT) = Unary(Sin, a) |> check
        static member Cos (a: ElemExprT) = Unary(Cos, a) |> check
        static member Tan (a: ElemExprT) = Unary(Tan, a) |> check
        static member Asin (a: ElemExprT) = Unary(Asin, a) |> check
        static member Acos (a: ElemExprT) = Unary(Acos, a) |> check
        static member Atan (a: ElemExprT) = Unary(Atan, a) |> check
        static member Sinh (a: ElemExprT) = Unary(Sinh, a) |> check
        static member Cosh (a: ElemExprT) = Unary(Cosh, a) |> check
        static member Tanh (a: ElemExprT) = Unary(Tanh, a) |> check
        static member Sqrt (a: ElemExprT) = Unary(Sqrt, a) |> check
        static member Ceiling (a: ElemExprT) = Unary(Ceil, a) |> check
        static member Floor (a: ElemExprT) = Unary(Floor, a) |> check
        static member Round (a: ElemExprT) = Unary(Round, a) |> check
        static member Truncate (a: ElemExprT) = Unary(Truncate, a) |> check

        // elementwise binary
        static member (+) (a: ElemExprT, b: ElemExprT) = Binary(Add, a, b) |> check
        static member (-) (a: ElemExprT, b: ElemExprT) = Binary(Substract, a, b) |> check
        static member (*) (a: ElemExprT, b: ElemExprT) = Binary(Multiply, a, b) |> check
        static member (/) (a: ElemExprT, b: ElemExprT) = Binary(Divide, a, b) |> check
        static member (%) (a: ElemExprT, b: ElemExprT) = Binary(Modulo, a, b) |> check
        static member Pow (a: ElemExprT, b: ElemExprT) = Binary(Power, a, b) |> check
        static member ( *** ) (a: ElemExprT, b: ElemExprT) = a ** b 

        // elementwise binary with basetype
        static member (+) (a: ElemExprT, b: 'T) = a + (scalar b) |> check
        static member (-) (a: ElemExprT, b: 'T) = a - (scalar b) |> check
        static member (*) (a: ElemExprT, b: 'T) = a * (scalar b) |> check
        static member (/) (a: ElemExprT, b: 'T) = a / (scalar b) |> check
        static member (%) (a: ElemExprT, b: 'T) = a % (scalar b) |> check
        static member Pow (a: ElemExprT, b: 'T) = a ** (scalar b) |> check
        static member ( *** ) (a: ElemExprT, b: 'T) = a ** (scalar b)

        static member (+) (a: 'T, b: ElemExprT) = (scalar a) + b |> check
        static member (-) (a: 'T, b: ElemExprT) = (scalar a) - b |> check
        static member (*) (a: 'T, b: ElemExprT) = (scalar a) * b |> check
        static member (/) (a: 'T, b: ElemExprT) = (scalar a) / b |> check
        static member (%) (a: 'T, b: ElemExprT) = (scalar a) % b |> check
        static member Pow (a: 'T, b: ElemExprT) = (scalar a) ** b |> check  
        static member ( *** ) (a: 'T, b: ElemExprT) = (scalar a) ** b          
          
    /// sign keeping type
    let signt (a: ElemExprT) =
        ElemExprT.SignT a |> check

    /// square root
    let sqrtt (a: ElemExprT) =
        ElemExprT.Sqrt a |> check
                  
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
            Unary (Sum (sumSym, first, last), expr) |> check
        | _ -> invalidArg "idx" "idx must be summation index obtained by calling sumIdx"

    /// If left=right, then thenExpr else elseExpr.
    let ifThenElse left right thenExpr elseExpr =
        Binary (IfThenElse (left, right), thenExpr, elseExpr) |> check

    /// expr if first <= sym <= last, otherwise 0.
    let kroneckerRng sym first last expr =
        Unary (KroneckerRng (sym, first, last), expr) |> check

    /// the element with index idx of the n-th argument
    [<RequiresExplicitTypeArguments>]
    let argElem<'T> pos idx =
        Leaf (ArgElement ((Arg pos, idx), TypeName.ofType<'T>)) |> check

    /// the element with index idx of the n-th argument of given type
    let argElemWithType typ pos idx =
        Leaf (ArgElement ((Arg pos, idx), TypeName.ofTypeInst typ)) |> check

    type Argument1D<'T> (pos: int) =       
        member this.Item with get (i0) : ElemExprT = argElem<'T> pos [i0]
    type Argument2D<'T> (pos: int) =       
        member this.Item with get (i0, i1) : ElemExprT = argElem<'T> pos [i0; i1]
    type Argument3D<'T> (pos: int) =       
        member this.Item with get (i0, i1, i2) : ElemExprT = argElem<'T> pos [i0; i1; i2]
    type Argument4D<'T> (pos: int) =       
        member this.Item with get (i0, i1, i2, i3) : ElemExprT = argElem<'T> pos [i0; i1; i2; i3]
    type Argument5D<'T> (pos: int) =       
        member this.Item with get (i0, i1, i2, i3, i4) : ElemExprT = argElem<'T> pos [i0; i1; i2; i3; i4]
    type Argument6D<'T> (pos: int) =       
        member this.Item with get (i0, i1, i2, i3, i4, i5) : ElemExprT = argElem<'T> pos [i0; i1; i2; i3; i4; i5]

    /// scalar argument at given position
    [<RequiresExplicitTypeArguments>] 
    let arg0D<'T> pos = argElem<'T> pos []
    /// 1-dimensional argument at given position
    [<RequiresExplicitTypeArguments>] 
    let arg1D<'T> pos = Argument1D<'T> pos
    /// 2-dimensional argument at given position
    [<RequiresExplicitTypeArguments>] 
    let arg2D<'T> pos = Argument2D<'T> pos
    /// 3-dimensional argument at given position
    [<RequiresExplicitTypeArguments>] 
    let arg3D<'T> pos = Argument3D<'T> pos
    /// 4-dimensional argument at given position
    [<RequiresExplicitTypeArguments>] 
    let arg4D<'T> pos = Argument4D<'T> pos
    /// 5-dimensional argument at given position
    [<RequiresExplicitTypeArguments>] 
    let arg5D<'T> pos = Argument5D<'T> pos
    /// 6-dimensional argument at given position
    [<RequiresExplicitTypeArguments>] 
    let arg6D<'T> pos = Argument6D<'T> pos

    /// extract ArgElementSpec from element expression
    let extractArg expr =
        match expr with
        | Leaf (ArgElement (argSpec, _)) -> argSpec
        | _ -> failwith "the provided element expression is not an argument"
   
    /// returns the required number of arguments of the element expression
    let rec requiredNumberOfArgs expr =
        match expr with
        | Leaf (ArgElement ((Arg n, _), _)) -> 
            if n < 0 then failwith "argument index must be positive"
            n

        | Leaf _ -> 0
        | Unary (op, a) -> requiredNumberOfArgs a
        | Binary (op, a, b) -> max (requiredNumberOfArgs a) (requiredNumberOfArgs b)
    
    /// checks if the arguments' shapes are compatible with the result shape and that the types match
    let checkCompatibility (expr: ElemExprT) (argShapes: ShapeSpecT list) (argTypes: TypeNameT list) 
            (resShape: ShapeSpecT) =

        // check number of arguments
        let nArgs = List.length argShapes
        let nReqArgs = requiredNumberOfArgs expr       
        if nReqArgs > nArgs then
            failwithf "the element expression requires at least %d arguments but only %d arguments were specified"
                nReqArgs nArgs

        // check dimensionality of arguments
        let rec check expr =
            match expr with
            | Leaf (ArgElement ((Arg n, idx), tn)) ->
                let idxDim = ShapeSpec.nDim idx
                let argDim = ShapeSpec.nDim argShapes.[n]
                if idxDim <> argDim then
                    failwithf 
                        "the argument with zero-based index %d has %d dimensions but was used  \
                         with %d dimensions in the element expression" n argDim idxDim
                let argType = argTypes.[n]
                if argType <> tn then
                    failwithf 
                        "the argument with zero-based index %d has type %A but was used  \
                         as type %A in the element expression" n argType.Type tn.Type
            | Leaf _ -> ()

            | Unary (_, a) -> check a
            | Binary (_, a, b) -> check a; check b
        check expr

    /// substitutes the specified size symbols with their replacements 
    let rec substSymSizes symSizes expr = 
        let sSub = substSymSizes symSizes
        let sSize = SymSizeEnv.subst symSizes
        let sShp = SymSizeEnv.substShape symSizes

        match expr with
        | Leaf (SizeValue (sc, tn)) -> Leaf (SizeValue ((sSize sc), tn))
        | Leaf (ArgElement ((arg, argIdxs), tn)) -> Leaf (ArgElement ((arg, sShp argIdxs), tn))
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
            | Leaf (SizeValue (sc, _)) -> SizeSpec.canEval sc
            | Leaf (ArgElement ((arg, argIdxs), _)) -> ShapeSpec.canEval argIdxs
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
    let rec prettyString (expr: ElemExprT) =
        // TODO: delete unnecessary brackets
        match expr with
        | Leaf (op) -> 
            match op with
            | Const v -> sprintf "%A" v
            | SizeValue (ss, _) -> sprintf "%A" ss
            | ArgElement ((Arg a, idxs), _) -> sprintf "a%d%A" a idxs
        
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
                sprintf "%s[%A%s%A %s %A%s%A](%s)"  (String('\u03B4',1)) 
                                                    first
                                                    (String('\u2264',1))
                                                    s
                                                    (String('\u2227',1))
                                                    s
                                                    (String('\u2264',1))
                                                    last
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

    type ElemExprT with
        member this.PrettyString = prettyString this


/// unified element expression
module UElemExpr = 
    open ElemExpr

    /// unified element expression op
    type UOpT =
        | ULeafOp of LeafOpT
        | UUnaryOp of UnaryOpT
        | UBinaryOp of BinaryOpT
    
    /// unified element expression
    type UElemExprT =
        | UElemExpr of UOpT * (UElemExprT list) * TypeNameT

    /// element function
    type UElemFuncT = {
        /// element expression
        Expr:       UElemExprT
        /// number of dimensions of the result
        NDims:      int
        /// number of input arguments
        NArgs:      int
    }

    /// converts an element expression to a unified element expression
    let rec toUElemExpr (elemExpr: ElemExprT) =
        let tn = ElemExpr.typeName elemExpr
        let leaf uop        = UElemExpr (ULeafOp uop, [], tn)
        let unary uop a     = UElemExpr (UUnaryOp uop, [toUElemExpr a], tn)
        let binary uop a b  = UElemExpr (UBinaryOp uop, [toUElemExpr a; toUElemExpr b], tn)

        match elemExpr with
        | Leaf op -> leaf op
        | Unary (op, a) -> unary op a
        | Binary (op, a, b) -> binary op a b           

    /// converts an element expression to a unified element function
    let toUElemFunc elemExpr nDims nArgs =
        {
            Expr    = toUElemExpr elemExpr
            NDims   = nDims
            NArgs   = nArgs
        }


[<AutoOpen>]
module ElemExprTypes =
    type ElemExprT = ElemExpr.ElemExprT
