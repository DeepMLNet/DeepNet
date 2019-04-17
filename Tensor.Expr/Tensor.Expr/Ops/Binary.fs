namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor.Expr
open Tensor


/// Addition.
type Add = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Add (ArgValue.binaryY argVals) |> Ch.only


/// Subtraction.
type Subtract = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Subtract (ArgValue.binaryY argVals) |> Ch.only


/// Multiplication.
type Multiply = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Multiply (ArgValue.binaryY argVals) |> Ch.only


/// Division.
type Divide = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Divide (ArgValue.binaryY argVals) |> Ch.only      


/// Exponentiation.
type Pow = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Pow (ArgValue.binaryY argVals) |> Ch.only


/// Modulo.
type Modulo = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Modulo (ArgValue.binaryY argVals) |> Ch.only       


/// Elementwise maximum.
type MaxElemwise = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).MaxElemwise (ArgValue.binaryY argVals) |> Ch.only


/// Elementwise minimum.
type MinElemwise = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).MinElemwise (ArgValue.binaryY argVals) |> Ch.only       


/// Element-wise if-then-else.
type IfThenElse = {Cond: BaseExprCh; IfTrue: BaseExprCh; IfFalse: BaseExprCh} with
    interface IOp with       
        member this.Check () = 
            Check.sameType [this.IfTrue; this.IfFalse]
            Check.sameDev [this.Cond; this.IfTrue; this.IfFalse]
            Check.bool [this.Cond]
            Check.sameShape [this.Cond; this.IfTrue; this.IfFalse]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.IfTrue.TypeName |> Ch.only
        member this.Devs = this.Cond.Dev |> Ch.only
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
        member this.Eval env argVals = 
            argVals.[Arg.Custom "IfTrue"].IfThenElse argVals.[Arg.Custom "IfFalse"] argVals.[Arg.Custom "Cond"]
            |> Ch.only


/// Logical And.
type And = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            (ArgValue.binaryX argVals :?> Tensor<bool>) &&&& (ArgValue.binaryY argVals :?> Tensor<bool>) :> ITensor       
            |> Ch.only


/// Logical Or.
type Or = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            (ArgValue.binaryX argVals :?> Tensor<bool>) |||| (ArgValue.binaryY argVals :?> Tensor<bool>) :> ITensor       
            |> Ch.only


/// Logical Xor.
type Xor = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            (ArgValue.binaryX argVals :?> Tensor<bool>) ^^^^ (ArgValue.binaryY argVals :?> Tensor<bool>) :> ITensor       
             |> Ch.only


/// Equal.
type Equal = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Equal (ArgValue.binaryY argVals) |> Ch.only       


/// Not equal.
type NotEqual = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).NotEqual (ArgValue.binaryY argVals) |> Ch.only       


/// Less than.
type Less = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Less (ArgValue.binaryY argVals) |> Ch.only       


/// Less then or equal.
type LessOrEqual = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).LessOrEqual (ArgValue.binaryY argVals) |> Ch.only       


/// Greater than.
type Greater = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Greater (ArgValue.binaryY argVals) |> Ch.only       


/// Greater than or equal.
type GreaterOrEqual = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameDev [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<bool> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).GreaterOrEqual (ArgValue.binaryY argVals) |> Ch.only


/// Dot product.
type Dot = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = 
            Check.sameType [this.X; this.Y]
            Check.sameDev [this.X; this.Y]
            let sa, sb = this.X.Shape, this.Y.Shape
            match Shape.nDim sa, Shape.nDim sb with
            | 2, 2 -> 
                if not (Size.equalIgnoringBc sa.[1] sb.[0]) then
                    failwithf "Incompatible shapes for dot product: %A and %A." sa sb
            | na, nb when na = nb -> 
                if not (Size.equalIgnoringBc sa.[na-1] sb.[nb-2]) || 
                    [0 .. na-3] |> List.exists (fun n -> not (Size.equalIgnoringBc sa.[n] sb.[n])) then
                        failwithf "Incompatible shapes for batched dot product: %A and %A." sa sb
            | _ -> failwithf "Cannot compute dot product between tensors of shapes %A and %A." sa sb  
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes =
            let sa, sb = this.X.Shape, this.Y.Shape
            match Shape.nDim sa, Shape.nDim sb with
            | 2, 2 -> Shape.matrix sa.[0] sb.[1]
            | na, nb when na=nb -> sa.[0 .. na-2] @ [sb.[nb-1]]
            | _ -> failwithf "Invalid dot product shapes: %A and %A." sa sb
            |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).Dot (ArgValue.binaryY argVals) |> Ch.only     


/// Tensor product.
type TensorProduct = { X: BaseExprCh; Y: BaseExprCh } with
    interface IOp with       
        member this.Check () = 
            Check.sameType [this.X; this.Y]
            Check.sameDev [this.X; this.Y]
            let sa, sb = this.X.Shape, this.Y.Shape
            if Shape.nDim sa <> Shape.nDim sb then
                failwithf "Cannot compute tensor product between tensors of shapes %A and %A." sa sb
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = 
            List.map2 (*) this.X.Shape this.Y.Shape |> Ch.only
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.binaryX argVals).TensorProduct (ArgValue.binaryY argVals) |> Ch.only      


/// Replace a slice of a tensor with another tensor.
type SetSubtensor = {X: BaseExprCh; Y: BaseExprCh; Range: SimpleRanges} with
    interface IOp with      
        member this.Check () = 
            Check.sameType [this.X; this.Y]
            Check.sameDev [this.X; this.Y]
            Check.range this.Range this.X
            if this.X.NDims <> this.Y.NDims then
                failwith "Source and target of SetSubtensor must be of same dimensionality."
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = 
            let xyArgs = Args.binary this.X this.Y
            let dynArgs = 
                SimpleRangesArgs.toArgs this.Range
                |> Map.map (fun _ v -> v :?> BaseExprCh)
            Map.join xyArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
            let range = this.Range |> SimpleRangesArgs.replaceFromArgs dynArgs               
            {this with X=Args.binaryX args; Y=Args.binaryY args; Range=range} :> _
        member this.SubstSymSizes env = {this with Range = SimpleRanges.subst env this.Range} :> _
        member this.CanEvalAllSymSizes = SimpleRanges.canEvalSymbols this.Range
        member this.Eval env argVals = 
            // TODO: dynamic range is always copied to host
            let dynVals = 
                argVals 
                |> Map.filter (fun arg _ -> 
                    match arg with
                    | Arg.N _ -> true
                    | _ -> false)
                |> Map.map (fun _ v -> Tensor.value (v :?> Tensor<int64>) |> Size.fix)
            let range = 
                this.Range 
                |> SimpleRangesArgs.resolveDynElems dynVals 
                |> SimpleRanges.eval
            let trgt = ArgValue.binaryX argVals |> ITensor.copy
            trgt.[range] <- ArgValue.binaryY argVals
            trgt |> Ch.only

    interface ITensorStubWishPropagatingOp with
        member this.PropagateWishes chWishes =
            // Wish for base tensor to be evaluated into channel wish.
            match chWishes |> Map.tryFind Ch.Default with
            | Some chWish -> Map [Arg.X, chWish]
            | None -> Map.empty

    interface IOpFormat with
        member this.Text =
            sprintf "SetSubtensor%A" this.Range
