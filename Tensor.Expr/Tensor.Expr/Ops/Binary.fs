namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor.Expr
open Tensor


/// Addition.
type Add = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Add (ArgValue.binaryY env.Args) |> Ch.only


/// Subtraction.
type Subtract = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Subtract (ArgValue.binaryY env.Args) |> Ch.only


/// Multiplication.
type Multiply = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Multiply (ArgValue.binaryY env.Args) |> Ch.only


/// Division.
type Divide = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Divide (ArgValue.binaryY env.Args) |> Ch.only      


/// Exponentiation.
type Pow = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Pow (ArgValue.binaryY env.Args) |> Ch.only


/// Modulo.
type Modulo = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Modulo (ArgValue.binaryY env.Args) |> Ch.only       


/// Elementwise maximum.
type MaxElemwise = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).MaxElemwise (ArgValue.binaryY env.Args) |> Ch.only


/// Elementwise minimum.
type MinElemwise = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).MinElemwise (ArgValue.binaryY env.Args) |> Ch.only       


/// Element-wise if-then-else.
type IfThenElse = {Cond: BaseExprCh; IfTrue: BaseExprCh; IfFalse: BaseExprCh} with
    interface IOp with       
        member this.Check () = 
            Check.sameType [this.IfTrue; this.IfFalse]
            Check.bool [this.Cond]
            Check.sameShape [this.Cond; this.IfTrue; this.IfFalse]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.IfTrue.TypeName |> Ch.only
        member this.Shapes = this.IfTrue.Shape |> Ch.only
        member this.Args = 
            Map [Arg.Custom "Cond", this.Cond
                 Arg.Custom "IfTrue", this.IfTrue
                 Arg.Custom "IfFalse", this.IfFalse]
        member this.ReplaceArgs args = 
            {this with Cond=args.[Arg.Custom "Cond"]
                       IfTrue=args.[Arg.Custom "IfTrue"]
                       IfFalse=args.[Arg.Custom "IfFalse"]} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            env.Args.[Arg.Custom "IfTrue"].IfThenElse env.Args.[Arg.Custom "IfFalse"] env.Args.[Arg.Custom "Cond"]
            |> Ch.only


/// Logical And.
type And = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            (ArgValue.binaryX env.Args :?> Tensor<bool>) &&&& (ArgValue.binaryY env.Args :?> Tensor<bool>) :> ITensor       
            |> Ch.only


/// Logical Or.
type Or = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            (ArgValue.binaryX env.Args :?> Tensor<bool>) |||| (ArgValue.binaryY env.Args :?> Tensor<bool>) :> ITensor       
            |> Ch.only


/// Logical Xor.
type Xor = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            (ArgValue.binaryX env.Args :?> Tensor<bool>) ^^^^ (ArgValue.binaryY env.Args :?> Tensor<bool>) :> ITensor       
             |> Ch.only


/// Equal.
type Equal = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Equal (ArgValue.binaryY env.Args) |> Ch.only       


/// Not equal.
type NotEqual = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).NotEqual (ArgValue.binaryY env.Args) |> Ch.only       


/// Less than.
type Less = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Less (ArgValue.binaryY env.Args) |> Ch.only       


/// Less then or equal.
type LessOrEqual = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).LessOrEqual (ArgValue.binaryY env.Args) |> Ch.only       


/// Greater than.
type Greater = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Greater (ArgValue.binaryY env.Args) |> Ch.only       


/// Greater than or equal.
type GreaterOrEqual = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).GreaterOrEqual (ArgValue.binaryY env.Args) |> Ch.only


/// Dot product.
type Dot = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
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
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes =
            let sa, sb = this.X.Shape, this.Y.Shape
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 2, 2 -> ShapeSpec.matrix sa.[0] sb.[1]
            | na, nb when na=nb -> sa.[0 .. na-2] @ [sb.[nb-1]]
            | _ -> failwithf "Invalid dot product shapes: %A and %A." sa sb
            |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).Dot (ArgValue.binaryY env.Args) |> Ch.only     


/// Tensor product.
type TensorProduct = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = 
            Check.sameType [this.X; this.Y]
            let sa, sb = this.X.Shape, this.Y.Shape
            if ShapeSpec.nDim sa <> ShapeSpec.nDim sb then
                failwithf "Cannot compute tensor product between tensors of shapes %A and %A." sa sb
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = 
            List.map2 (*) this.X.Shape this.Y.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (ArgValue.binaryX env.Args).TensorProduct (ArgValue.binaryY env.Args) |> Ch.only      


/// Replace a slice of a tensor with another tensor.
type SetSubtensor = {X: BaseExprCh; Y: BaseExprCh; Range: SimpleRangesSpec} with
    interface IOp with      
        member this.Check () = 
            Check.sameType [this.X; this.Y]
            Check.range this.Range this.X
            if this.X.NDims <> this.Y.NDims then
                failwith "Source and target of SetSubtensor must be of same dimensionality."
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = 
            let xyArgs = Args.binary this.X this.Y
            let dynArgs = 
                SimpleRangesSpecArgs.toArgs this.Range
                |> Map.map (fun _ v -> v :?> BaseExprCh)
            Map.join xyArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
            let range = this.Range |> SimpleRangesSpecArgs.replaceFromArgs dynArgs               
            {this with X=Args.binaryX args; Y=Args.binaryY args; Range=range} :> _
        member this.SubstSymSizes env = {this with Range = SymSizeEnv.substRange env this.Range} :> _
        member this.CanEvalAllSymSizes = SimpleRangesSpec.canEvalSymbols this.Range
        member this.Eval env = 
            // TODO: dynamic range is always copied to host
            let dynVals = 
                env.Args 
                |> Map.filter (fun arg _ -> 
                    match arg with
                    | Arg.N _ -> true
                    | _ -> false)
                |> Map.map (fun _ v -> Tensor.value (v :?> Tensor<int64>) |> SizeSpec.fix)
            let range = 
                this.Range 
                |> SimpleRangesSpecArgs.resolveDynElems dynVals 
                |> SimpleRangesSpec.eval
            let trgt = ArgValue.binaryX env.Args |> ITensor.copy
            trgt.[range] <- ArgValue.binaryY env.Args
            trgt |> Ch.only


