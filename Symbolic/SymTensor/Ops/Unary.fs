namespace SymTensor.Ops

open DeepNet.Utils
open SymTensor
open Tensor
open OpTools


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
type Abs = { X: BaseExpr } with
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
type SignT = { X: BaseExpr } with
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
type Log = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Log ()       


/// Logarithm to base 10.
type Log10 = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Log10 ()       


/// Exponential function.
type Exp = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Exp ()       


/// Sine.
type Sin = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Sin ()       


/// Cosine.
type Cos = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Cos ()       


/// Tangent.
type Tan = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Tan ()       


/// Inverse sine.
type Asin = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true

        member this.Eval env = (Args.unaryX env.Args).Asin ()       


/// Inverse cosine.
type Acos = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Acos ()       


/// Inverse tangent.
type Atan = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Atan ()       


/// Hyperbolic sine.
type Sinh = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Sinh ()       

/// Hyperbolic cosine.
type Cosh = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Cosh ()       

/// Hyperbolic tangent.
type Tanh = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Tanh ()       
        
/// Square root.
type Sqrt = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Sqrt ()       

/// Round towards positive infinity.
type Ceiling = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Ceiling ()       

/// Round towards negative infinity.
type Floor = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Floor ()       

/// Round towards nearest integer.
type Round = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Round ()       

/// Round towards zeros.
type Truncate = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).Truncate ()       

/// (Batched) matrix inverse.
type Invert = { X: BaseExpr } with
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
        member this.Eval env = (Args.unaryX env.Args).Invert ()


/// Logical not.
type Not = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = Check.bool [this.X]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = ~~~~(Args.unaryX env.Args :?> Tensor<bool>) :> ITensor       

/// Reshape
type Reshape = { X: BaseExpr; Shape: ShapeSpec } with
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
        member this.Eval env =
            (Args.unaryX env.Args) |> ITensor.reshape (ShapeSpec.eval this.Shape)       

/// Broadcast.
type DoBroadcast = { X: BaseExpr; Shape: ShapeSpec } with
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
        member this.Eval env = (Args.unaryX env.Args) |> ITensor.broadcastTo (ShapeSpec.eval this.Shape)      


/// Permute the axes.
type PermuteAxes = {X: BaseExpr; Permutation: int list} with
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
        member this.Eval env = (Args.unaryX env.Args) |> ITensor.permuteAxes this.Permutation


/// Read a slice from a tensor.
type Subtensor = {X: BaseExpr; Range: SimpleRangesSpec} with
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
                |> Map.map (fun _ v -> v :?> BaseExpr)
            Map.join xArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
            let range = this.Range |> SimpleRangesSpec.replaceDynElems dynPrefix dynArgs               
            {this with X=Args.unaryX args; Range=range} :> _
        member this.SubstSymSizes env = {this with Range = SymSizeEnv.substRange env this.Range} :> _
        member this.CanEvalAllSymSizes = SimpleRangesSpec.canEvalSymbols this.Range
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


/// Reverses the tensor in the specified dimension.
type ReverseAxis = {X: BaseExpr; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape 
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args) |> ITensor.reverseAxis this.Axis


/// Extract the diagonal(s) along the given axes.
type Diag = {X: BaseExpr; Axis1: int; Axis2: int} with
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
        member this.Eval env = (Args.unaryX env.Args).DiagAxis this.Axis1 this.Axis2


/// Build a matrix with the specified diagonal.
type DiagMat = {X: BaseExpr; Axis1: int; Axis2: int} with
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
        member this.Eval env = (Args.unaryX env.Args).DiagMatAxis this.Axis1 this.Axis2


/// Sum over specified axis.
type SumAxis = {X: BaseExpr; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).SumAxis this.Axis 


/// Product over specified axis.
type ProductAxis = {X: BaseExpr; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).ProductAxis this.Axis


/// Maximum over specified axis.
type MaxAxis = {X: BaseExpr; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).MaxAxis this.Axis


/// Minimum over specified axis.
type MinAxis = {X: BaseExpr; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).MinAxis this.Axis


/// Maximum over specified axis.
type ArgMaxAxis = {X: BaseExpr; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = TypeName.ofType<int64>
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).ArgMaxAxis this.Axis


/// Minimum over specified axis.
type ArgMinAxis = {X: BaseExpr; Axis: int} with
    interface IOp2 with      
        member this.Check () = Check.axis this.Axis this.X
        member this.TypeName = TypeName.ofType<int64>
        member this.Shape = this.X.Shape |> ShapeSpec.withoutAxis this.Axis
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.unaryX env.Args).ArgMinAxis this.Axis


/// Select elements according to the specified index tensors
type Gather = {X: BaseExpr; Indices: BaseExpr option list} with
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
                                (this.Indices |> List.map (Option.map (fun e -> e.Shape)))
                | None when dim >= ShapeSpec.nDim trgtShape ->
                    failwithf "Gather index dimensions beyond the number of target dimensions \
                                must not be None."
                | _ -> ()
        member this.TypeName = this.X.TypeName
        member this.Shape = (this.Indices |> List.pick id).Shape
        member this.Args = 
            let idxArgs = this.Indices |> listToMap                
            let xArgs = Args.unary this.X
            Map.join idxArgs xArgs
        member this.ReplaceArgs args =                
            {this with X=Args.unaryX args; Indices=mapToList this.Indices.Length args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            let vIndices = env.Args |> mapToList this.Indices.Length
            (Args.unaryX env.Args).Gather vIndices 


/// Disperses elements according to the specified index tensors.
type Scatter = {X: BaseExpr; Indices: BaseExpr option list; Shape: ShapeSpec} with
    interface IOp2 with      
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
        member this.Eval env = 
            let vIndices = env.Args |> mapToList this.Indices.Length
            (Args.unaryX env.Args).Scatter vIndices (ShapeSpec.eval this.Shape)


/// Store value to variable.
type Store = {X: BaseExpr; Var: Var} with
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
        member this.Eval env = 
            let tv = env.VarEnv |> VarEnv.get this.Var 
            let v = Args.unaryX env.Args                
            tv.CopyFrom (v.Transfer tv.Dev)
            v.ZerosOfSameType v.Dev [0L]


/// Sets the Jacobian of its argument to zero when calculating derivatives.
type AssumeZeroDeriv = { X: BaseExpr } with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = Args.unaryX env.Args
    

/// Sets the Jacobian of its argument to zero when calculating derivatives.
type AssumeDeriv = {Deriv: BaseExpr; X: BaseExpr} with
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
        member this.Eval env = Args.unaryX env.Args
    

/// Annotation (no influence on value).
type Annotated = {Label: System.IComparable; X: BaseExpr} with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = Args.unaryX env.Args                  

    
/// Prints the value together with the given label.
type Print = {Label: string; X: BaseExpr} with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            let v = Args.unaryX env.Args
            printfn "%s=\n%A\n" this.Label v
            v                            
    

/// Dumps the result into the given dataset in the active HDF5 dump file.
type Dump = {Dataset: string; X: BaseExpr} with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            let v = Args.unaryX env.Args
            Dump.dumpValue this.Dataset v
            v                            


/// If the value contains NaNs or infinities, outputs their location and 
/// stops the computation.
type CheckFinite = {Label: string; X: BaseExpr} with
    interface IOp2 with      
        member this.Check () = ()
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            let v = Args.unaryX env.Args
            if not (v.AllFinite ()) then
                printfn "Infinity or NaN encountered in %s with value:\n%A" this.Label v
                failwithf "Infinity or NaN encountered in %s." this.Label
            v                            


/// Accesses the specified channel of a multi-channnel expression.
type Channel = {Channel: string; X: BaseMultiChannelExpr} with
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
        member this.Eval env = 
            let v = Args.unaryX env.MultiChannelArgs
            v.[this.Channel]
    interface IMultiChannelArgsOp with
        member this.MultiChannelArgs = Args.unary this.X
        member this.ReplaceMultiChannelArgs args = {this with X=Args.unaryX args} :> _


