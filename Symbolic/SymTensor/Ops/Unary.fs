namespace SymTensor.Ops

open DeepNet.Utils
open SymTensor


/// Unary plus.
type UnaryPlus = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).UnaryPlus ()      


/// Negation.
type Negate = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).UnaryMinus ()       


/// Absolute value.
type Abs = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Abs ()       

    
/// Sign.
type SignT = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Sgn ()       


/// Logarithm to base exp.
type Log = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            let one = Deriv.one this.X
            dOp * Expr2.padLeft (this.X ** (-one)) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Log ()       
let (|Log|_|) (expr: Expr2) =
    match expr.Op with
    | :? Log as this -> Some this.X
    | _ -> None

/// Logarithm to base 10.
type Log10 = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            let one = Deriv.one this.X
            let ten = Deriv.ten this.X
            dOp * Expr2.padLeft (this.X ** (-one) / log ten) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Log10 ()       
let (|Log10|_|) (expr: Expr2) =
    match expr.Op with
    | :? Log10 as this -> Some this.X
    | _ -> None

/// Exponential function.
type Exp = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (exp this.X) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Exp ()       
let (|Exp|_|) (expr: Expr2) =
    match expr.Op with
    | :? Exp as this -> Some this.X
    | _ -> None

/// Sine.
type Sin = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (cos this.X) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Sin ()       
let (|Sin|_|) (expr: Expr2) =
    match expr.Op with
    | :? Sin as this -> Some this.X
    | _ -> None

/// Cosine.
type Cos = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (-sin this.X) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Cos ()       
let (|Cos|_|) (expr: Expr2) =
    match expr.Op with
    | :? Cos as this -> Some this.X
    | _ -> None

/// Tangent.
type Tan = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            let one = Deriv.one this.X
            let two = Deriv.two this.X
            dOp * Expr2.padLeft (one + (tan this.X) ** two) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Tan ()       
let (|Tan|_|) (expr: Expr2) =
    match expr.Op with
    | :? Tan as this -> Some this.X
    | _ -> None

/// Inverse sine.
type Asin = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            let one = Deriv.one this.X
            let two = Deriv.two this.X
            dOp * Expr2.padLeft (one / Expr2.sqrtt (one - this.X ** two)) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Asin ()       
let (|Asin|_|) (expr: Expr2) =
    match expr.Op with
    | :? Asin as this -> Some this.X
    | _ -> None

/// Inverse cosine.
type Acos = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp =
            let one = Deriv.one this.X
            let two = Deriv.two this.X
            dOp * Expr2.padLeft (-one / Expr2.sqrtt (one - this.X ** two)) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Acos ()       
let (|Acos|_|) (expr: Expr2) =
    match expr.Op with
    | :? Acos as this -> Some this.X
    | _ -> None

/// Inverse tangent.
type Atan = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp =
            let one = Deriv.one this.X
            let two = Deriv.two this.X
            dOp * Expr2.padLeft (one / (one + this.X ** two)) |> Args.unary 
        member this.Eval env = (Args.unaryX env.Args).Atan ()       
let (|Atan|_|) (expr: Expr2) =
    match expr.Op with
    | :? Atan as this -> Some this.X
    | _ -> None

/// Hyperbolic sine.
type Sinh = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (cosh this.X) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Sinh ()       
let (|Sinh|_|) (expr: Expr2) =
    match expr.Op with
    | :? Sinh as this -> Some this.X
    | _ -> None

/// Hyperbolic cosine.
type Cosh = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (sinh this.X) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Cosh ()       
let (|Cosh|_|) (expr: Expr2) =
    match expr.Op with
    | :? Cosh as this -> Some this.X
    | _ -> None

/// Hyperbolic tangent.
type Tanh = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            let one = Deriv.one this.X
            let two = Deriv.two this.X
            dOp * Expr2.padLeft (one - (tanh this.X) ** two) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Tanh ()       
let (|Tanh|_|) (expr: Expr2) =
    match expr.Op with
    | :? Tanh as this -> Some this.X
    | _ -> None
        
/// Square root.
type Sqrt = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            let one = Deriv.one this.X
            let two = Deriv.two this.X
            dOp * Expr2.padLeft (one / (two * Expr2.sqrtt this.X)) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Sqrt ()       
let (|Sqrt|_|) (expr: Expr2) =
    match expr.Op with
    | :? Sqrt as this -> Some this.X
    | _ -> None

/// Round towards positive infinity.
type Ceiling = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Deriv.zeros dOp this.X |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Ceiling ()       
let (|Ceiling|_|) (expr: Expr2) =
    match expr.Op with
    | :? Ceiling as this -> Some this.X
    | _ -> None

/// Round towards negative infinity.
type Floor = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Deriv.zeros dOp this.X |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Floor ()       
let (|Floor|_|) (expr: Expr2) =
    match expr.Op with
    | :? Floor as this -> Some this.X
    | _ -> None

/// Round towards nearest integer.
type Round = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Deriv.zeros dOp this.X |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Round ()       
let (|Round|_|) (expr: Expr2) =
    match expr.Op with
    | :? Round as this -> Some this.X
    | _ -> None

/// Round towards zeros.
type Truncate = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Deriv.zeros dOp this.X |> Args.unary
        member this.Eval env = (Args.unaryX env.Args).Truncate ()       
let (|Truncate|_|) (expr: Expr2) =
    match expr.Op with
    | :? Truncate as this -> Some this.X
    | _ -> None

/// (Batched) matrix inverse.
type Invert = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = 
            if this.X.NDims < 2 then
                failwithf "Need at least a matrix to invert but got shape %A" this.X.Shape
            if this.X.Shape.[this.X.NDims-2] .<> this.X.Shape.[this.X.NDims-1] then
                failwithf "Cannot invert non-square matrix %A along last two axes." this.X.Shape
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = 
            let self = this |> Expr2
            -(Expr2.padLeft self.T) .* dOp .* (Expr2.padLeft self.T) |> Args.unary 
        member this.Eval env = (Args.unaryX env.Args).Invert ()
let (|Invert|_|) (expr: Expr2) =
    match expr.Op with
    | :? Invert as this -> Some this.X
    | _ -> None

/// Computes the inverse of a matrix.
/// If the input has more than two dimensions, the inverses
/// along the last two dimensions are returned.
/// The inverse of a singular matrix is undefinied.
/// No error is raised in that case.
let invert (x: Expr2) =
    {Invert.X=x} |> Expr2

/// Logical not.
type Not = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = Check.bool [this.X]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Deriv.zeros dOp this.X |> Args.unary
        member this.Eval env = ~~~~(Args.unaryX env.Args :?> Tensor<bool>) :> ITensor       
let (|Not|_|) (expr: Expr2) =
    match expr.Op with
    | :? Not as this -> Some this.X
    | _ -> None

/// Reshape
type Reshape = { X: Expr2; Shape: ShapeSpec } with
    interface IOp2 with      
        member this.Check () = 
            if ShapeSpec.nElem this.X.Shape .<> ShapeSpec.nElem this.Shape then
                failwithf "Cannot change number of elements while reshaping from %A to %A." 
                            this.X.Shape this.Shape
        member this.TypeName = this.X.TypeName
        member this.Shape = this.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = 
            { this with Shape = SymSizeEnv.substShape env this.Shape } :> _
        member this.CanEvalAllSymSizes = 
            ShapeSpec.canEval this.Shape
        member this.Deriv dOp =
            let funElems = dOp.Shape.[0]
            dOp |> Expr2.reshape (funElems :: this.X.Shape) |> Args.unary
        member this.Eval env =
            (Args.unaryX env.Args) |> ITensor.reshape (ShapeSpec.eval this.Shape)       
let (|Reshape|_|) (expr: Expr2) =
    match expr.Op with
    | :? Reshape as this -> Some this
    | _ -> None

/// Broadcast.
type DoBroadcast = { X: Expr2; Shape: ShapeSpec } with
    interface IOp2 with      
        member this.Check () = 
            if ShapeSpec.nDim this.X.Shape <> ShapeSpec.nDim this.Shape then
                failwithf "Tensor of shape %A does not have same number of dimesions as broadcast shape %A."
                            this.X.Shape this.Shape
            for dim in 0 .. (ShapeSpec.nDim this.Shape) - 1 do
                match this.X.Shape.[dim], this.Shape.[dim] with
                | SizeSpec.Broadcast, _ -> ()
                | ssa, ssb when ssa .<> ssb -> 
                    failwithf "Cannot broadcast from %A to %A because non-broadcast dimensions must not change." 
                                this.X.Shape this.Shape
                | _ -> ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = 
            { this with Shape = SymSizeEnv.substShape env this.Shape } :> _
        member this.CanEvalAllSymSizes = 
            ShapeSpec.canEval this.Shape
        member this.Deriv dOp = 
            let mutable dOpUnBc = dOp
            for ax, (bSize, xSize) in List.indexed (List.zip this.Shape this.X.Shape) do
                match bSize, xSize with
                | SizeSpec.Broadcast, SizeSpec.Broadcast -> ()
                | _, SizeSpec.Broadcast ->
                    dOpUnBc <- dOpUnBc |> sumKeepingAxis (ax + 1)
                | _ -> ()
            dOpUnBc |> Args.unary                 
        member this.Eval env = (Args.unaryX env.Args) |> ITensor.broadcastTo (ShapeSpec.eval this.Shape)      
let (|DoBroadcast|_|) (expr: Expr2) =
    match expr.Op with
    | :? DoBroadcast as this -> Some this
    | _ -> None

/// Permute the axes.
type PermuteAxes = {X: Expr2; Permutation: int list} with
    interface IOp2 with      
        member this.Check () = 
            if ShapeSpec.nDim this.X.Shape <> List.length this.Permutation then
                failwithf "Permutation %A must have same rank as shape %A." this.Permutation this.X.Shape
            if not (Permutation.is this.Permutation) then
                failwithf "%A is not a valid permutation of an %d-dimensional tensor." 
                            this.Permutation (ShapeSpec.nDim this.X.Shape)
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.permuteAxes this.Permutation
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp =
            let backPerm = Permutation.invert this.Permutation
            let dOpPerm = 
                0 :: List.map (fun p -> p + 1) backPerm
            dOp |> Expr2.permuteAxes dOpPerm |> Args.unary                 
        member this.Eval env = (Args.unaryX env.Args) |> ITensor.permuteAxes this.Permutation
let (|PermuteAxes|_|) (expr: Expr2) =
    match expr.Op with
    | :? PermuteAxes as this -> Some this
    | _ -> None

let private dynPrefix = "D"


/// Read a slice from a tensor.
type Subtensor = {X: Expr2; Range: SimpleRangesSpec} with
    interface IOp2 with      
        member this.Check () = 
            Check.range this.Range this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = 
            (this.Range, this.X.Shape)
            ||> List.map2 (fun sr shp ->
                match sr with
                | SimpleRangeSpec.SymStartSymEnd (s, fo)    -> (fo |? (shp - SizeSpec.one)) + 1L - s
                | SimpleRangeSpec.DynStartSymSize (_, size) -> size)            
        member this.Args = 
            let xArgs = Args.unary this.X 
            let dynArgs = 
                SimpleRangesSpec.dynElems dynPrefix this.Range
                |> Map.map (fun _ v -> v :?> Expr2)
            Map.join xArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
            let range = this.Range |> SimpleRangesSpec.replaceDynElems dynPrefix dynArgs               
            {this with X=Args.unaryX args; Range=range} :> _
        member this.SubstSymSizes env = {this with Range = SymSizeEnv.substRange env this.Range} :> _
        member this.CanEvalAllSymSizes = SimpleRangesSpec.canEvalSymbols this.Range
        member this.Deriv dOp = 
            let funElems = dOp.Shape.[0]
            let agExpanded = Expr2.zerosOfType dOp.DataType (funElems :: this.X.Shape)
            Expr2.setSubtensor agExpanded.[SimpleRangeSpec.All :: this.Range] dOp
            |> Args.unary
        member this.Eval env = 
            // TODO: dynamic range is always copied to host
            let dynVals = 
                env.Args 
                |> Map.filter (fun k _ -> k.StartsWith dynPrefix)
                |> Map.map (fun _ v -> Tensor.value (v :?> Tensor<int64>) |> SizeSpec.fix)
            let range = 
                this.Range 
                |> SimpleRangesSpec.resolveDynElems dynPrefix dynVals 
                |> SimpleRangesSpec.eval
            (Args.unaryX env.Args).[range]
let (|Subtensor|_|) (expr: Expr2) =
    match expr.Op with
    | :? Subtensor as this -> Some this
    | _ -> None

/// Replace a slice of a tensor with another tensor.
type SetSubtensor = {X: Expr2; Y: Expr2; Range: SimpleRangesSpec} with
    interface IOp2 with      
        member this.Check () = 
            Check.sameType [this.X; this.Y]
            Check.range this.Range this.X
            if this.X.NDims <> this.Y.NDims then
                failwith "Source and target of SetSubtensor must be of same dimensionality."
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape           
        member this.Args = 
            let xyArgs = Args.binary this.X this.Y
            let dynArgs = 
                SimpleRangesSpec.dynElems dynPrefix this.Range
                |> Map.map (fun _ v -> v :?> Expr2)
            Map.join xyArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
            let range = this.Range |> SimpleRangesSpec.replaceDynElems dynPrefix dynArgs               
            {this with X=Args.binaryX args; Y=Args.binaryY args; Range=range} :> _
        member this.SubstSymSizes env = {this with Range = SymSizeEnv.substRange env this.Range} :> _
        member this.CanEvalAllSymSizes = SimpleRangesSpec.canEvalSymbols this.Range
        member this.Deriv dOp = 
            let dYExp = dOp.[SimpleRangeSpec.All :: this.Range]
            let zeros = Expr2.zerosOfType dYExp.DataType dYExp.Shape
            let dXExp = Expr2.setSubtensor dOp.[SimpleRangeSpec.All :: this.Range] zeros
            Args.binary dXExp dYExp
        member this.Eval env = 
            // TODO: dynamic range is always copied to host
            let dynVals = 
                env.Args 
                |> Map.filter (fun k _ -> k.StartsWith dynPrefix)
                |> Map.map (fun _ v -> Tensor.value (v :?> Tensor<int64>) |> SizeSpec.fix)
            let range = 
                this.Range 
                |> SimpleRangesSpec.resolveDynElems dynPrefix dynVals 
                |> SimpleRangesSpec.eval
            let trgt = Args.binaryX env.Args |> ITensor.copy
            trgt.[range] <- Args.binaryY env.Args
            trgt
let (|SetSubtensor|_|) (expr: Expr2) =
    match expr.Op with
    | :? SetSubtensor as this -> Some this
    | _ -> None

let internal isSubtensor (expr: Expr2) =
    match expr with
    | Reshape {X=Subtensor {Range=range; X=trgtExpr} as subtensorExpr} ->
        Some (range, subtensorExpr, trgtExpr)
    | _ -> None

/// Reverses the tensor in the specified dimension.
type ReverseAxis = {X: Expr2; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape 
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp =
            dOp |> reverseAxis (this.Axis + 1) |> Args.unary
        member this.Eval env = (Args.unaryX env.Args) |> ITensor.reverseAxis this.Axis
let (|ReverseAxis|_|) (expr: Expr2) =
    match expr.Op with
    | :? ReverseAxis as this -> Some this
    | _ -> None   

/// Reverses the tensor in the specified dimension.
let reverseAxis axis (x: Expr2) =
    {ReverseAxis.Axis=axis; X=x} |> Expr2

/// Extract the diagonal(s) along the given axes.
type Diag = {X: Expr2; Axis1: int; Axis2: int} with
    interface IOp2 with      
        member this.Check () = 
            Check.axis this.Axis1 this.X
            Check.axis this.Axis2 this.X 
            if not (this.Axis1 < this.Axis2) then 
                failwith "First axis for extracting diagonal must come before second axis."
            if this.X.Shape.[this.Axis1] .<> this.X.Shape.[this.Axis2] then
                failwithf "Cannot extract diagonal along axes %d and %d from non-square tensor with shape %A" 
                            this.Axis1 this.Axis2 this.X.Shape
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis2
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = (Args.unaryX env.Args).DiagAxis this.Axis1 this.Axis2
let (|Diag|_|) (expr: Expr2) =
    match expr.Op with
    | :? Diag as this -> Some this
    | _ -> None   

/// Extracts the diagonal along the given axes.
let diagAxis ax1 ax2 (x: Expr2) = 
    let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
    {Diag.Axis1=ax1; Axis2=ax2; Diag.X=x} |> Expr2
                             
/// Extracts the diagonal of a matrix.
/// If the expression has more than two dimensions, the diagonals
/// are extracted along the last two dimensions.
let diag (x: Expr2) = 
    if x.NDims < 2 then 
        failwithf "Need at least a matrix to extract diagonal but got shape: %A" x.Shape
    x |> diagAxis (x.NDims-2) (x.NDims-1)

/// Build a matrix with the specified diagonal.
type DiagMat = {X: Expr2; Axis1: int; Axis2: int} with
    interface IOp2 with      
        member this.Check () = 
            Check.axis this.Axis1 this.X
            if not (0 <= this.Axis2 && this.Axis2 <= this.X.NDims) then
                failwithf "Cannot build diagonal matrix over non-existant axis %d of tensor with shape %A." 
                            this.Axis2 this.X.Shape
            if not (this.Axis1 < this.Axis2) then 
                failwith "First axis for building diagonal matrix must come before second axis."
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> List.insert this.Axis2 this.X.Shape.[this.Axis1]
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = (Args.unaryX env.Args).DiagMatAxis this.Axis1 this.Axis2
let (|DiagMat|_|) (expr: Expr2) =
    match expr.Op with
    | :? DiagMat as this -> Some this
    | _ -> None   

/// Creates a diagonal matrix by duplicating the given dimension.
let diagMatAxis ax1 ax2 (x: Expr2) = 
    let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
    {DiagMat.Axis1=ax1; Axis2=ax2; X=x} |> Expr2

/// Creates a matrix with the given vector on its diagonal. 
/// All other elements are zeros.
/// If the input has more than one dimension, the operation is
/// performed batch-wise on the last dimension.
let diagMat (x: Expr2) =
    if x.NDims < 1 then 
        failwithf "Need at least a vector to build diagonal matrix but got shape: %A" x.Shape
    x |> diagMatAxis (x.NDims-1) x.NDims

/// Sum over specified axis.
type SumAxis = {X: Expr2; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = (Args.unaryX env.Args).SumAxis this.Axis 
let (|SumAxis|_|) (expr: Expr2) =
    match expr.Op with
    | :? SumAxis as this -> Some this
    | _ -> None

/// summation over given dimension
let sumAxis axis x = 
    {SumAxis.Axis=axis; X=x} |> Expr2

/// summation over given dimension, while keeping the axis with one (broadcastable) element
let sumKeepingAxis axis x =
    x |> sumAxis axis |> Expr2.insertBroadcastAxis axis

/// summaiton of all elements
let sum x = 
    x |> Expr2.flatten |> sumAxis 0

/// Computes the traces along the given axes.
let traceAxis ax1 ax2 x =
    let tax = if ax1 < ax2 then ax1 else ax1 + 1
    x |> diagAxis ax1 ax2 |> sumAxis tax

/// Computes the trace of a matrix.
/// If the input has more than two dimensions, the traces
/// along the last two dimensions are returned.
let trace (x: Expr2) =
    if x.NDims < 2 then
        failwithf "Need at least a matrix for trace but got shape: %A" x.Shape      
    x |> traceAxis (x.NDims-2) (x.NDims-1) 

/// Product over specified axis.
type ProductAxis = {X: Expr2; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = (Args.unaryX env.Args).ProductAxis this.Axis
let (|ProductAxis|_|) (expr: Expr2) =
    match expr.Op with
    | :? ProductAxis as this -> Some this
    | _ -> None

/// product over given dimension
let productAxis axis x = 
    {ProductAxis.Axis=axis; X=x} |> Expr2

/// product over given dimension, while keeping the axis with one (broadcastable) element
let productKeepingAxis axis x =
    x |> productAxis axis |> Expr2.insertBroadcastAxis axis

/// product of all elements
let product x = 
    x |> Expr2.flatten |> productAxis 0

/// Maximum over specified axis.
type MaxAxis = {X: Expr2; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = (Args.unaryX env.Args).MaxAxis this.Axis
let (|MaxAxis|_|) (expr: Expr2) =
    match expr.Op with
    | :? MaxAxis as this -> Some this
    | _ -> None

/// Maximum over given dimension.
let maxAxis axis x = 
    {MaxAxis.Axis=axis; X=x} |> Expr2

/// Maximum over given dimension, while keeping the axis with one (broadcastable) element.
let maxKeepingAxis axis x =
    x |> maxAxis axis |> Expr2.insertBroadcastAxis axis

/// Maximum of all elements.
let max x = 
    x |> Expr2.flatten |> maxAxis 0

/// Minimum over specified axis.
type MinAxis = {X: Expr2; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = (Args.unaryX env.Args).MinAxis this.Axis
let (|MinAxis|_|) (expr: Expr2) =
    match expr.Op with
    | :? MinAxis as this -> Some this
    | _ -> None

/// Minimum over given dimension.
let minAxis axis x = 
    {MinAxis.Axis=axis; X=x} |> Expr2

/// Minimum over given dimension, while keeping the axis with one (broadcastable) element.
let minKeepingAxis axis x =
    x |> minAxis axis |> Expr2.insertBroadcastAxis axis

/// Minimum of all elements.
let min x = 
    x |> Expr2.flatten |> minAxis 0

/// Maximum over specified axis.
type ArgMaxAxis = {X: Expr2; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = TypeName.ofType<int64>
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = (Args.unaryX env.Args).ArgMaxAxis this.Axis
let (|ArgMaxAxis|_|) (expr: Expr2) =
    match expr.Op with
    | :? ArgMaxAxis as this -> Some this
    | _ -> None

/// Index of maximum over given dimension.
let argMaxAxis axis x = 
    {ArgMaxAxis.Axis=axis; X=x} |> Expr2

/// Index of maximum over given dimension, while keeping the axis with one (broadcastable) element.
let argMaxKeepingAxis axis x =
    x |> argMaxAxis axis |> Expr2.insertBroadcastAxis axis

/// Minimum over specified axis.
type ArgMinAxis = {X: Expr2; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = TypeName.ofType<int64>
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = (Args.unaryX env.Args).ArgMinAxis this.Axis
let (|ArgMinAxis|_|) (expr: Expr2) =
    match expr.Op with
    | :? ArgMinAxis as this -> Some this
    | _ -> None

/// Index of minimum over given dimension.
let argMinAxis axis x = 
    {MinAxis.Axis=axis; X=x} |> Expr2

/// Index of minimum over given dimension, while keeping the axis with one (broadcastable) element.
let argMinKeepingAxis axis x =
    x |> minAxis axis |> Expr2.insertBroadcastAxis axis

let private listToMap (list: 'a option list) =
    list 
    |> List.indexed 
    |> List.choose (function 
                    | i, Some v -> Some (sprintf "I%d" i, v)
                    | _, None -> None)
    |> Map.ofList

let private mapToList length (map: Map<string, 'a>) =
    [0 .. length-1]
    |> List.map (fun i -> map |> Map.tryFind (sprintf "I%d" i))

/// Select elements according to the specified index tensors
type Gather = {X: Expr2; Indices: Expr2 option list} with
    interface IOp2 with      
        member this.Check () = 
            if this.X.NDims <> this.Indices.Length then
                failwithf "Gather source has shape %A but %d index tensors were specified." 
                            this.X.Shape this.Indices.Length
            let trgtShape =
                match this.Indices |> List.tryPick id with
                | Some idx -> idx.Shape
                | None -> failwith "Gather needs at least one specified (not None) index expression."  
            for dim, idx in List.indexed this.Indices do
                match idx with
                | Some idx when idx.DataType <> typeof<int64> ->
                    failwithf "All index tensors for gather must be of type int64, but got type %A." idx.DataType
                | Some idx when idx.Shape <> trgtShape ->
                    failwithf "All gather indices must have equal shape, but got shapes %A."
                                (this.Indices |> List.map (Option.map Expr2.shape))
                | None when dim >= ShapeSpec.nDim trgtShape ->
                    failwithf "Gather index dimensions beyond the number of target dimensions \
                                must not be None."
                | _ -> ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.Indices |> List.pick id |> Expr2.shape
        member this.Args = 
            let idxArgs = this.Indices |> listToMap                
            let xArgs = Args.unary this.X
            Map.join idxArgs xArgs
        member this.ReplaceArgs args =                
            {this with X=Args.unaryX args; Indices=mapToList this.Indices.Length args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = 
            let vIndices = env.Args |> mapToList this.Indices.Length
            (Args.unaryX env.Args).Gather vIndices 
let (|Gather|_|) (expr: Expr2) =
    match expr.Op with
    | :? Gather as this -> Some this
    | _ -> None    

/// Select elements according to the specified index tensors.
let gather (indices: Expr2 option list) (x: Expr2) =
    let someIndices = indices |> List.choose id
    if List.isEmpty someIndices then
        failwith "Gather needs at least one specified index tensor."
    let bcSomeIndices = Expr2.broadcastToSameMany someIndices
    let rec rebuild idxs repIdxs =
        match idxs, repIdxs with
        | Some idx :: rIdxs, repIdx :: rRepIdxs ->
            Some repIdx :: rebuild rIdxs rRepIdxs
        | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
        | [], [] -> []
        | _ -> failwith "unbalanced idxs"
    let bcIndices = rebuild indices bcSomeIndices
    {Gather.Indices=bcIndices; X=x} |> Expr2

/// Disperses elements according to the specified index tensors.
type Scatter = {X: Expr2; Indices: Expr2 option list; Shape: ShapeSpec} with
    interface IOp2 with      
        member this.Check () = 
            for dim, idx in List.indexed this.Indices do
                match idx with
                | Some idx when idx.DataType <> typeof<int64> ->
                    failwithf "All index tensors for scatter must be of type int64, but got type %A." idx.DataType
                | Some idx when idx.Shape <> this.X.Shape ->
                    failwithf "All scatter indices must have shape of source %A, but got %A." 
                                this.X.Shape (this.Indices |> List.map (Option.map Expr2.shape))
                | None when dim >= this.X.NDims ->
                    failwithf "Scatter index dimensions beyond the number of source dimensions \
                                must not be None."
                | _ -> ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.Shape
        member this.Args = 
            let idxArgs = this.Indices |> listToMap                
            let xArgs = Args.unary this.X
            Map.join idxArgs xArgs
        member this.ReplaceArgs args =                
            {this with X=Args.unaryX args; Indices=mapToList this.Indices.Length args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary -dOp // TODO
        member this.Eval env = 
            let vIndices = env.Args |> mapToList this.Indices.Length
            (Args.unaryX env.Args).Scatter vIndices (ShapeSpec.eval this.Shape)
let (|Scatter|_|) (expr: Expr2) =
    match expr.Op with
    | :? Scatter as this -> Some this
    | _ -> None   

/// Disperses elements according to the specified index tensors.
let scatter (indices: Expr2 option list) (trgtShp: ShapeSpec) (x: Expr2) =
    let indices = indices |> List.map (Option.map (Expr2.broadcastToShape x.Shape))
    {Scatter.Indices=indices; Shape=trgtShp; X=x} |> Expr2

/// Store value to variable.
type Store = {X: Expr2; Var: Var} with
    interface IOp2 with       
        member this.Check () = 
            if this.X.TypeName <> this.Var.TypeName then
                failwithf "Cannot store expression of type %A into variable of type %A."
                            this.X.TypeName this.Var.TypeName
            if not (ShapeSpec.equalWithoutBroadcastability this.X.Shape this.Var.Shape) then
                failwithf "Cannot store expression of shape %A into variable of shape %A." 
                            this.X.Shape this.Var.Shape                
        member this.TypeName = this.X.TypeName
        member this.Shape = ShapeSpec.emptyVector
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = 
            {this with X=Args.unaryX args} :> _
        member this.SubstSymSizes env = 
            {this with Var={this.Var with Shape=SymSizeEnv.substShape env this.Var.Shape}} :> _
        member this.CanEvalAllSymSizes = ShapeSpec.canEval this.Var.Shape
        member this.Deriv dOp = Map.empty
        member this.Eval env = 
            let tv = env.VarEnv |> VarEnv.get this.Var 
            let v = Args.unaryX env.Args                
            tv.CopyFrom (v.Transfer tv.Dev)
            v.ZerosOfSameType v.Dev [0L]
let (|Store|_|) (expr: Expr2) =
    match expr.Op with
    | :? Store as this -> Some this
    | _ -> None

/// Sets the Jacobian of its argument to zero when calculating derivatives.
type AssumeZeroDeriv = { X: Expr2 } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary dOp // TODO
        member this.Eval env = Args.unaryX env.Args
let (|AssumeZeroDeriv|_|) (expr: Expr2) =
    match expr.Op with
    | :? AssumeZeroDeriv as this -> Some this.X
    | _ -> None
    
/// Nullifies the Jacobian of its argument when calculating derivatives.
let assumeZeroDeriv x =
    {AssumeZeroDeriv.X=x} |> Expr2

/// Sets the Jacobian of its argument to zero when calculating derivatives.
type AssumeDeriv = {Deriv: Expr2; X: Expr2} with
    interface IOp2 with      
        member this.Check () = 
            Check.sameType [this.Deriv; this.X]
            if this.Deriv.NDims <> 2 then
                failwithf "Jacobian shape %A must be two-dimensional." this.Deriv.Shape
            if this.Deriv.Shape.[1] <> this.X.NElems then
                failwithf "Jacobian shape %A must have %A elements in second dimension." 
                    this.Deriv.Shape this.X.NElems
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args =                 
            Map.join (Args.unary this.X) (Map ["Deriv", this.Deriv])                
        member this.ReplaceArgs args = 
            {this with Deriv=args.["Deriv"]; X=Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary dOp // TODO
        member this.Eval env = Args.unaryX env.Args
let (|AssumeDeriv|_|) (expr: Expr2) =
    match expr.Op with
    | :? AssumeDeriv as this -> Some this
    | _ -> None
    
/// Assumes the specified Jacobian when calculating derivatives.
let assumeDeriv deriv x =
    {AssumeDeriv.Deriv=deriv; X=x} |> Expr2

/// Annotation (no influence on value).
type Annotated = {Label: System.IComparable; X: Expr2} with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary dOp 
        member this.Eval env = Args.unaryX env.Args                  
let (|Annotated|_|) (expr: Expr2) =
    match expr.Op with
    | :? Annotated as this -> Some this
    | _ -> None

/// Annotated expression (no influence on value).
let annotate label x = 
    {Annotated.Label=label; X=x} |> Expr2
    
/// Prints the value together with the given label.
type Print = {Label: string; X: Expr2} with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary dOp 
        member this.Eval env = 
            let v = Args.unaryX env.Args
            printfn "%s=\n%A\n" this.Label v
            v                            
let (|Print|_|) (expr: Expr2) =
    match expr.Op with
    | :? Print as this -> Some this
    | _ -> None
    
/// Print the result with the given label when evaluated.
let print label x =
    {Print.Label=label; X=x} |> Expr2

/// Dumps the result into the given dataset in the active HDF5 dump file.
type Dump = {Dataset: string; X: Expr2} with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary dOp 
        member this.Eval env = 
            let v = Args.unaryX env.Args
            Dump.dumpValue this.Dataset v
            v                            
let (|Dump|_|) (expr: Expr2) =
    match expr.Op with
    | :? Dump as this -> Some this
    | _ -> None

/// Dumps the result into the given dataset in the active HDF5 dump file.
let dump dataset x =
    {Dump.Dataset=dataset; X=x} |> Expr2

/// If the value contains NaNs or infinities, outputs their location and 
/// stops the computation.
type CheckFinite = {Label: string; X: Expr2} with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary dOp 
        member this.Eval env = 
            let v = Args.unaryX env.Args
            if not (v.AllFinite ()) then
                printfn "Infinity or NaN encountered in %s with value:\n%A" this.Label v
                failwithf "Infinity or NaN encountered in %s." this.Label
            v                            
let (|CheckFinite|_|) (expr: Expr2) =
    match expr.Op with
    | :? CheckFinite as this -> Some this
    | _ -> None

/// If the value contains NaNs or infinities, outputs their location and 
/// stops the computation.
let checkFinite label x =
    {CheckFinite.Label=label; X=x} |> Expr2

/// Accesses the specified channel of a multi-channnel expression.
type Channel = {Channel: string; X: MultiChannelExpr} with
    interface IOp2 with      
        member this.Check () = 
            if not (this.X.Channels |> List.contains this.Channel) then
                failwithf "Multi-channel expression with channels %A does not have channel %A." 
                            this.X.Channels this.Channel 
        member this.TypeName = this.X.TypeNames.[this.Channel]
        member this.Shape = this.X.Shapes.[this.Channel]
        member this.Args = Map.empty
        member this.ReplaceArgs args = this :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.unary dOp 
        member this.Eval env = 
            let v = Args.unaryX env.MultiChannelArgs
            v.[this.Channel]
    interface IMultiChannelArgsOp with
        member this.MultiChannelArgs = Args.unary this.X
        member this.ReplaceMultiChannelArgs args = {this with X=Args.unaryX args} :> _
let (|Channel|_|) (expr: Expr2) =
    match expr.Op with
    | :? Channel as this -> Some this
    | _ -> None
