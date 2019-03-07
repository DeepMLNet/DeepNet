namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor.Expr
open Tensor


/// Unary plus.
type UnaryPlus = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).UnaryPlus () |> Ch.only     


/// Negation.
type Negate = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).UnaryMinus () |> Ch.only      


/// Absolute value.
type Abs = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Abs () |> Ch.only     

    
/// Sign.
type SignT = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only 
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Sgn () |> Ch.only


/// Logarithm to base exp.
type Log = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Log () |> Ch.only


/// Logarithm to base 10.
type Log10 = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Log10 () |> Ch.only


/// Exponential function.
type Exp = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Exp () |> Ch.only


/// Sine.
type Sin = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Sin () |> Ch.only


/// Cosine.
type Cos = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Cos () |> Ch.only


/// Tangent.
type Tan = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Tan () |> Ch.only


/// Inverse sine.
type Asin = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Asin () |> Ch.only


/// Inverse cosine.
type Acos = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Acos () |> Ch.only       


/// Inverse tangent.
type Atan = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Atan () |> Ch.only


/// Hyperbolic sine.
type Sinh = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Sinh () |> Ch.only      


/// Hyperbolic cosine.
type Cosh = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Cosh () |> Ch.only      


/// Hyperbolic tangent.
type Tanh = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Tanh () |> Ch.only      
        

/// Square root.
type Sqrt = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Sqrt () |> Ch.only     


/// Round towards positive infinity.
type Ceiling = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Ceiling () |> Ch.only      


/// Round towards negative infinity.
type Floor = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Floor () |> Ch.only


/// Round towards nearest integer.
type Round = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Round () |> Ch.only


/// Round towards zeros.
type Truncate = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Truncate () |> Ch.only


/// (Batched) matrix inverse.
type Invert = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = 
            if this.X.NDims < 2 then
                failwithf "Need at least a matrix to invert but got shape %A" this.X.Shape
            if this.X.Shape.[this.X.NDims-2] .<> this.X.Shape.[this.X.NDims-1] then
                failwithf "Cannot invert non-square matrix %A along last two axes." this.X.Shape
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).Invert () |> Ch.only


/// Logical not.
type Not = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = Check.bool [this.X]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = ~~~~(ArgValue.unaryX argVals :?> Tensor<bool>) :> ITensor |> Ch.only


/// Reshape
type Reshape = { X: BaseExprCh; Shape: ShapeSpec } with
    interface IOp with      
        member this.Check () = 
            if ShapeSpec.nElem this.X.Shape .<> ShapeSpec.nElem this.Shape then
                failwithf "Cannot change number of elements while reshaping from %A to %A." 
                            this.X.Shape this.Shape
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = 
            { this with Shape = SymSizeEnv.substShape env this.Shape } :> _
        member this.CanEvalAllSymSizes = 
            ShapeSpec.canEval this.Shape
        member this.Eval env argVals =
            (ArgValue.unaryX argVals) |> ITensor.reshape (ShapeSpec.eval this.Shape) |> Ch.only
    interface IOpFormat with
        member this.Text =
            sprintf "Reshape%A" this.Shape

/// Broadcast.
type DoBroadcast = { X: BaseExprCh; Shape: ShapeSpec } with
    interface IOp with      
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
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = 
            { this with Shape = SymSizeEnv.substShape env this.Shape } :> _
        member this.CanEvalAllSymSizes = 
            ShapeSpec.canEval this.Shape
        member this.Eval env argVals = (ArgValue.unaryX argVals) |> ITensor.broadcastTo (ShapeSpec.eval this.Shape) |> Ch.only
    interface IOpFormat with
        member this.Text =
            sprintf "DoBroadcast%A" this.Shape

/// Permute the axes.
type PermuteAxes = {X: BaseExprCh; Permutation: int list} with
    interface IOp with      
        member this.Check () = 
            if ShapeSpec.nDim this.X.Shape <> List.length this.Permutation then
                failwithf "Permutation %A must have same rank as shape %A." this.Permutation this.X.Shape
            if not (Permutation.is this.Permutation) then
                failwithf "%A is not a valid permutation of an %d-dimensional tensor." 
                            this.Permutation (ShapeSpec.nDim this.X.Shape)
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> ShapeSpec.permuteAxes this.Permutation |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals) |> ITensor.permuteAxes this.Permutation |> Ch.only
    interface IOpFormat with
        member this.Text =
            sprintf "PermuteAxes%A" this.Permutation

/// Read a slice from a tensor.
type Subtensor = {X: BaseExprCh; Range: SimpleRangesSpec} with
    interface IOp with      
        member this.Check () = 
            Check.range this.Range this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = 
            (this.Range, this.X.Shape)
            ||> List.map2 (fun sr shp ->
                match sr with
                | SimpleRangeSpec.SymStartSymEnd (s, fo)    -> (fo |? (shp - SizeSpec.one)) + 1L - s
                | SimpleRangeSpec.DynStartSymSize (_, size) -> size)            
            |> Ch.only
        member this.Args = 
            let xArgs = Args.unary this.X 
            let dynArgs = 
                SimpleRangesSpecArgs.toArgs this.Range
                |> Map.map (fun _ v -> v :?> BaseExprCh)
            Map.join xArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
            let range = this.Range |> SimpleRangesSpecArgs.replaceFromArgs dynArgs               
            {this with X=Args.unaryX args; Range=range} :> _
        member this.SubstSymSizes env = {this with Range = SymSizeEnv.substRange env this.Range} :> _
        member this.CanEvalAllSymSizes = SimpleRangesSpec.canEvalSymbols this.Range
        member this.Eval env argVals = 
            // TODO: dynamic range is always copied to host
            let dynVals = 
                argVals 
                |> Map.filter (fun arg _ -> 
                    match arg with
                    | Arg.N _ -> true
                    | _ -> false)
                |> Map.map (fun _ v -> Tensor.value (v :?> Tensor<int64>) |> SizeSpec.fix)
            let range = 
                this.Range 
                |> SimpleRangesSpecArgs.resolveDynElems dynVals 
                |> SimpleRangesSpec.eval
            (ArgValue.unaryX argVals).[range] |> Ch.only


/// Reverses the tensor in the specified dimension.
type ReverseAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals) |> ITensor.reverseAxis this.Axis |> Ch.only


/// Extract the diagonal(s) along the given axes.
type Diag = {X: BaseExprCh; Axis1: int; Axis2: int} with
    interface IOp with      
        member this.Check () = 
            Check.axis this.Axis1 this.X
            Check.axis this.Axis2 this.X 
            if not (this.Axis1 < this.Axis2) then 
                failwith "First axis for extracting diagonal must come before second axis."
            if this.X.Shape.[this.Axis1] .<> this.X.Shape.[this.Axis2] then
                failwithf "Cannot extract diagonal along axes %d and %d from non-square tensor with shape %A" 
                            this.Axis1 this.Axis2 this.X.Shape
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> ShapeSpec.withoutAxis this.Axis2 |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).DiagAxis this.Axis1 this.Axis2 |> Ch.only


/// Build a matrix with the specified diagonal.
type DiagMat = {X: BaseExprCh; Axis1: int; Axis2: int} with
    interface IOp with      
        member this.Check () = 
            Check.axis this.Axis1 this.X
            if not (0 <= this.Axis2 && this.Axis2 <= this.X.NDims) then
                failwithf "Cannot build diagonal matrix over non-existant axis %d of tensor with shape %A." 
                            this.Axis2 this.X.Shape
            if not (this.Axis1 < this.Axis2) then 
                failwith "First axis for building diagonal matrix must come before second axis."
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> List.insert this.Axis2 this.X.Shape.[this.Axis1] |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).DiagMatAxis this.Axis1 this.Axis2 |> Ch.only


/// Sum over specified axis.
type SumAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> ShapeSpec.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).SumAxis this.Axis |> Ch.only


/// Product over specified axis.
type ProductAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> ShapeSpec.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).ProductAxis this.Axis |> Ch.only


/// Maximum over specified axis.
type MaxAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> ShapeSpec.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).MaxAxis this.Axis |> Ch.only


/// Minimum over specified axis.
type MinAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> ShapeSpec.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).MinAxis this.Axis |> Ch.only


/// Maximum over specified axis.
type ArgMaxAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int64> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> ShapeSpec.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).ArgMaxAxis this.Axis |> Ch.only


/// Minimum over specified axis.
type ArgMinAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int64> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> ShapeSpec.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).ArgMinAxis this.Axis |> Ch.only


/// Select elements according to the specified index tensors
type Gather = {X: BaseExprCh; Indices: BaseExprCh option list} with
    interface IOp with      
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
                                (this.Indices |> List.map (Option.map (fun e -> e.Shape)))
                | None when dim >= ShapeSpec.nDim trgtShape ->
                    failwithf "Gather index dimensions beyond the number of target dimensions \
                                must not be None."
                | _ -> ()
        member this.Channels = Ch.onlyOne   
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = (this.Indices |> List.pick id).Shape |> Ch.only
        member this.Args = 
            let idxArgs = Args.naryOpt this.Indices
            let xArgs = Args.unary this.X
            Map.join idxArgs xArgs
        member this.ReplaceArgs args =                
            {this with 
                X = Args.unaryX args
                Indices = Args.naryOptXs this.Indices.Length args
            } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let vIndices = argVals |> ArgValue.naryOptXs this.Indices.Length
            (ArgValue.unaryX argVals).Gather vIndices  |> Ch.only


/// Disperses elements according to the specified index tensors.
type Scatter = {X: BaseExprCh; Indices: BaseExprCh option list; Shape: ShapeSpec} with
    interface IOp with      
        member this.Check () = 
            for dim, idx in List.indexed this.Indices do
                match idx with
                | Some idx when idx.DataType <> typeof<int64> ->
                    failwithf "All index tensors for scatter must be of type int64, but got type %A." idx.DataType
                | Some idx when idx.Shape <> this.X.Shape ->
                    failwithf "All scatter indices must have shape of source %A, but got %A." 
                                this.X.Shape (this.Indices |> List.map (Option.map (fun e -> e.Shape)))
                | None when dim >= this.X.NDims ->
                    failwithf "Scatter index dimensions beyond the number of source dimensions \
                                must not be None."
                | _ -> ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.Shape |> Ch.only
        member this.Args = 
            let idxArgs = Args.naryOpt this.Indices            
            let xArgs = Args.unary this.X
            Map.join idxArgs xArgs
        member this.ReplaceArgs args =                
            {this with 
                X = Args.unaryX args
                Indices = Args.naryOptXs this.Indices.Length args
            } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let vIndices = argVals |> ArgValue.naryOptXs this.Indices.Length
            (ArgValue.unaryX argVals).Scatter vIndices (ShapeSpec.eval this.Shape) |> Ch.only


/// Store value to variable.
type Store = {X: BaseExprCh; Var: Var} with
    interface IOp with       
        member this.Check () = 
            if this.X.TypeName <> this.Var.TypeName then
                failwithf "Cannot store expression of type %A into variable of type %A."
                            this.X.TypeName this.Var.TypeName
            if not (ShapeSpec.equalWithoutBroadcastability this.X.Shape this.Var.Shape) then
                failwithf "Cannot store expression of shape %A into variable of shape %A." 
                            this.X.Shape this.Var.Shape   
        member this.Channels = Ch.onlyOne                            
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = ShapeSpec.emptyVector |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = 
            {this with X=Args.unaryX args} :> _
        member this.SubstSymSizes env = 
            {this with Var={this.Var with Shape=SymSizeEnv.substShape env this.Var.Shape}} :> _
        member this.CanEvalAllSymSizes = ShapeSpec.canEval this.Var.Shape
        member this.Eval env argVals = 
            let tv = env.VarEnv.[this.Var.Name]
            let v = ArgValue.unaryX argVals                
            tv.CopyFrom (v.Transfer tv.Dev)
            v.ZerosOfSameType v.Dev [0L] |> Ch.only


/// Sets the Jacobian of its argument to zero when calculating derivatives.
type AssumeZeroDeriv = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = ArgValue.unaryX argVals |> Ch.only
    

/// Sets the Jacobian of its argument to zero when calculating derivatives.
type AssumeDeriv = {Deriv: BaseExprCh; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = 
            Check.sameType [this.Deriv; this.X]
            if this.Deriv.NDims <> 2 then
                failwithf "Jacobian shape %A must be two-dimensional." this.Deriv.Shape
            if this.Deriv.Shape.[1] <> this.X.NElems then
                failwithf "Jacobian shape %A must have %A elements in second dimension." 
                    this.Deriv.Shape this.X.NElems
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args =                 
            Map.join (Args.unary this.X) (Map [Arg.Custom "Deriv", this.Deriv])                
        member this.ReplaceArgs args = 
            {this with 
                Deriv = args.[Arg.Custom "Deriv"] 
                X = Args.unaryX args
            } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = ArgValue.unaryX argVals |> Ch.only
    

/// Annotation (no influence on value).
type Annotated = {Label: System.IComparable; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = ArgValue.unaryX argVals |> Ch.only                 

    
/// Prints the value together with the given label.
type Print = {Label: string; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            printfn "%s=\n%A\n" this.Label v
            v |> Ch.only                          
    

/// Dumps the result into the given dataset in the active HDF5 dump file.
type Dump = {Dataset: string; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            Dump.dumpValue this.Dataset v
            v |> Ch.only                            


/// If the value contains NaNs or infinities, outputs their location and 
/// stops the computation.
type CheckFinite = {Label: string; X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            if not (v.AllFinite ()) then
                printfn "Infinity or NaN encountered in %s with value:\n%A" this.Label v
                failwithf "Infinity or NaN encountered in %s." this.Label
            v |> Ch.only                            


/// Returns the specified channel of a multi-channnel as its only channel.
type Channel = {X: BaseExprCh} with
    interface IOp with      
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X=Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            ArgValue.unaryX argVals |> Ch.only



