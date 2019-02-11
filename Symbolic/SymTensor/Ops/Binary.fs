namespace SymTensor.Ops

open SymTensor

/// Addition.
type Add = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Add (Args.binaryY env.Args)      
let (|Add|_|) (expr: Expr2) =
    match expr.Op with
    | :? Add as this -> Some (this.X, this.Y)
    | _ -> None

/// Subtraction.
type Subtract = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Subtract (Args.binaryY env.Args)       
let (|Subtract|_|) (expr: Expr2) =
    match expr.Op with
    | :? Subtract as this -> Some (this.X, this.Y)
    | _ -> None

/// Multiplication.
type Multiply = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Multiply (Args.binaryY env.Args)      
let (|Multiply|_|) (expr: Expr2) =
    match expr.Op with
    | :? Multiply as this -> Some (this.X, this.Y)
    | _ -> None

/// Division.
type Divide = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Divide (Args.binaryY env.Args)       
let (|Divide|_|) (expr: Expr2) =
    match expr.Op with
    | :? Divide as this -> Some (this.X, this.Y)
    | _ -> None

/// Exponentiation.
type Pow = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Pow (Args.binaryY env.Args)       
let (|Pow|_|) (expr: Expr2) =
    match expr.Op with
    | :? Pow as this -> Some (this.X, this.Y)
    | _ -> None

/// Modulo.
type Modulo = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Modulo (Args.binaryY env.Args)       
let (|Modulo|_|) (expr: Expr2) =
    match expr.Op with
    | :? Modulo as this -> Some (this.X, this.Y)
    | _ -> None

/// Elementwise maximum.
type MaxElemwise = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).MaxElemwise (Args.binaryY env.Args)       
let (|MaxElemwise|_|) (expr: Expr2) =
    match expr.Op with
    | :? MaxElemwise as this -> Some (this.X, this.Y)
    | _ -> None

/// Elementwise minimum.
type MinElemwise = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).MinElemwise (Args.binaryY env.Args)       
let (|MinElemwise|_|) (expr: Expr2) =
    match expr.Op with
    | :? MinElemwise as this -> Some (this.X, this.Y)
    | _ -> None

/// Logical And.
type And = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = 
            (Args.binaryX env.Args :?> Tensor<bool>) &&&& (Args.binaryY env.Args :?> Tensor<bool>) :> ITensor       
let (|And|_|) (expr: Expr2) =
    match expr.Op with
    | :? And as this -> Some (this.X, this.Y)
    | _ -> None

/// Logical Or.
type Or = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = 
            (Args.binaryX env.Args :?> Tensor<bool>) |||| (Args.binaryY env.Args :?> Tensor<bool>) :> ITensor       
let (|Or|_|) (expr: Expr2) =
    match expr.Op with
    | :? Or as this -> Some (this.X, this.Y)
    | _ -> None

/// Logical Xor.
type Xor = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = 
            (Args.binaryX env.Args :?> Tensor<bool>) ^^^^ (Args.binaryY env.Args :?> Tensor<bool>) :> ITensor       
let (|Xor|_|) (expr: Expr2) =
    match expr.Op with
    | :? Xor as this -> Some (this.X, this.Y)
    | _ -> None

/// Equal.
type Equal = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Equal (Args.binaryY env.Args)       
let (|Equal|_|) (expr: Expr2) =
    match expr.Op with
    | :? Equal as this -> Some (this.X, this.Y)
    | _ -> None

/// Not equal.
type NotEqual = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).NotEqual (Args.binaryY env.Args)       
let (|NotEqual|_|) (expr: Expr2) =
    match expr.Op with
    | :? NotEqual as this -> Some (this.X, this.Y)
    | _ -> None

/// Less than.
type Less = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Less (Args.binaryY env.Args)       
let (|Less|_|) (expr: Expr2) =
    match expr.Op with
    | :? Less as this -> Some (this.X, this.Y)
    | _ -> None

/// Less then or equal.
type LessOrEqual = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).LessOrEqual (Args.binaryY env.Args)       
let (|LessOrEqual|_|) (expr: Expr2) =
    match expr.Op with
    | :? LessOrEqual as this -> Some (this.X, this.Y)
    | _ -> None

/// Greater than.
type Greater = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Greater (Args.binaryY env.Args)       
let (|Greater|_|) (expr: Expr2) =
    match expr.Op with
    | :? Greater as this -> Some (this.X, this.Y)
    | _ -> None

/// Greater than or equal.
type GreaterOrEqual = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).GreaterOrEqual (Args.binaryY env.Args)       
let (|GreaterOrEqual|_|) (expr: Expr2) =
    match expr.Op with
    | :? GreaterOrEqual as this -> Some (this.X, this.Y)
    | _ -> None

/// Dot product.
type Dot = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = 
            Check.sameType [this.X; this.Y]
            let sa, sb = this.X.Shape, this.Y.Shape
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 2, 2 -> 
                if sa.[1] .<> sb.[0] then
                    failwithf "Incompatible shapes for dot product: %A and %A." sa sb
            | na, nb when na = nb -> 
                if sa.[na-1] .<> sb.[nb-2] || 
                    [0 .. na-3] |> List.exists (fun n -> sa.[n] .<> sb.[n]) then
                        failwithf "Incompatible shapes for batched dot product: %A and %A." sa sb
            | _ -> failwithf "Cannot compute dot product between tensors of shapes %A and %A." sa sb  
        member this.TypeName = this.X.TypeName
        member this.Shape =
            let sa, sb = this.X.Shape, this.Y.Shape
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 2, 2 -> ShapeSpec.matrix sa.[0] sb.[1]
            | na, nb when na=nb -> sa.[0 .. na-2] @ [sb.[nb-1]]
            | _ -> failwithf "Invalid dot product shapes: %A and %A." sa sb
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).Dot (Args.binaryY env.Args)       
let (|Dot|_|) (expr: Expr2) =
    match expr.Op with
    | :? Dot as this -> Some (this.X, this.Y)
    | _ -> None

/// Dot product.
/// Behavior depends on the dimensionality of the arguments.
/// Cases: 
/// (1, 1) -> vector-vector dot product resulting in a scalar
/// (2, 1) -> matrix-vector dot product resulting in a vector
/// (2, 2) -> matrix-matrix dot product resulting in a matrix
/// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
/// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
let dot (a: Expr2) (b: Expr2) =
    let sa, sb = a.Shape, b.Shape
    match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
    | 1, 1 -> 
        // vector-vector dot product
        sum (a * b)
    | 2, 1 -> 
        // matrix-vector dot product
        let bm = b |> Expr2.reshape (ShapeSpec.padRight sb)
        {Dot.X=a; Y=bm} |> Expr2 |> Expr2.reshape [sa.[0]]
    | 2, 2 -> 
        // matrix-matrix dot product
        {Dot.X=a; Y=b} |> Expr2
    | na, nb when na = nb -> 
        // batched matrix-matrix dot product
        let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa sb
        let ba = a |> Expr2.broadcast bsa
        let bb = b |> Expr2.broadcast bsb    
        {Dot.X=ba; Y=bb} |> Expr2
    | na, nb when na = nb + 1 ->
        // batched matrix-vector dot product
        let psb = ShapeSpec.padRight sb
        let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa psb
        let ba = a |> Expr2.broadcast bsa
        let bb = b |> Expr2.reshape psb |> Expr2.broadcast bsb    
        {Dot.X=ba; Y=bb} |> Expr2 |> Expr2.reshape bsa.[0 .. na-2]
    | _ -> failwithf "Cannot compute dot product between tensors of shapes %A and %A." sa sb  

/// Tensor product.
type TensorProduct = { X: Expr2; Y: Expr2 } with
    interface IOp2 with       
        member this.Check () = 
            Check.sameType [this.X; this.Y]
            let sa, sb = this.X.Shape, this.Y.Shape
            if ShapeSpec.nDim sa <> ShapeSpec.nDim sb then
                failwithf "Cannot compute tensor product between tensors of shapes %A and %A." sa sb
        member this.TypeName = this.X.TypeName
        member this.Shape = 
            List.map2 (*) this.X.Shape this.Y.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = (Args.binaryX env.Args).TensorProduct (Args.binaryY env.Args)       
let (|TensorProduct|_|) (expr: Expr2) =
    match expr.Op with
    | :? TensorProduct as this -> Some (this.X, this.Y)
    | _ -> None

let tensorProduct (x: Expr2) (y: Expr2) =
    {TensorProduct.X=x; Y=y} |> Expr2


/// Element-wise if-then-else.
type IfThenElse = {Cond: Expr2; IfTrue: Expr2; IfFalse: Expr2} with
    interface IOp2 with       
        member this.Check () = 
            Check.sameType [this.IfTrue; this.IfFalse]
            Check.bool [this.Cond]
            Check.sameShape [this.Cond; this.IfTrue; this.IfFalse]
        member this.TypeName = this.IfTrue.TypeName
        member this.Shape = this.IfTrue.Shape
        member this.Args = 
            Map ["Cond", this.Cond
                    "IfTrue", this.IfTrue
                    "IfFalse", this.IfFalse]
        member this.ReplaceArgs args = 
            {this with Cond=args.["Cond"]
                        IfTrue=args.["IfTrue"]
                        IfFalse=args.["IfFalse"]} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp
        member this.Eval env = 
            env.Args.["IfTrue"].IfThenElse env.Args.["IfFalse"] env.Args.["Cond"]
let (|IfThenElse|_|) (expr: Expr2) =
    match expr.Op with
    | :? IfThenElse as this -> Some this
    | _ -> None

/// Elementwise uses elements from `ifTrue` if `cond` is true for that element, otherwise elements from `ifFalse`.
let ifThenElse (cond: Expr2) (ifTrue: Expr2) (ifFalse: Expr2) =
    let shps = [cond.Shape; ifTrue.Shape; ifFalse.Shape]
    let pShps = ShapeSpec.padToSameMany shps
    let bcShps = ShapeSpec.broadcastToSameMany false pShps           
    match pShps, bcShps with
    | [condPShp; ifTruePShp; ifFalsePShp], [condBcShp; ifTrueBcShp; ifFalseBcShp] -> 
        let condBc = cond |> Expr2.reshape condPShp |> Expr2.broadcast condBcShp
        let ifTrueBc = ifTrue |> Expr2.reshape ifTruePShp |> Expr2.broadcast ifTrueBcShp
        let ifFalseBc = ifFalse |> Expr2.reshape ifFalsePShp |> Expr2.broadcast ifFalseBcShp
        {IfThenElse.Cond=condBc; IfTrue=ifTrueBc; IfFalse=ifFalseBc} |> Expr2
    | _ -> failwith "impossible"
