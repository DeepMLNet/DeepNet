namespace SymTensor.Ops

open DeepNet.Utils
open SymTensor
open Tensor
open OpTools


/// Addition.
type Add = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).Add (Args.binaryY env.Args)      


/// Subtraction.
type Subtract = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).Subtract (Args.binaryY env.Args)       


/// Multiplication.
type Multiply = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).Multiply (Args.binaryY env.Args)      


/// Division.
type Divide = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).Divide (Args.binaryY env.Args)       


/// Exponentiation.
type Pow = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).Pow (Args.binaryY env.Args)       


/// Modulo.
type Modulo = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).Modulo (Args.binaryY env.Args)       


/// Elementwise maximum.
type MaxElemwise = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).MaxElemwise (Args.binaryY env.Args)       


/// Elementwise minimum.
type MinElemwise = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).MinElemwise (Args.binaryY env.Args)       


/// Element-wise if-then-else.
type IfThenElse = {Cond: BaseExpr; IfTrue: BaseExpr; IfFalse: BaseExpr} with
    interface IOp with       
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
        member this.Eval env = 
            env.Args.["IfTrue"].IfThenElse env.Args.["IfFalse"] env.Args.["Cond"]


/// Logical And.
type And = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            (Args.binaryX env.Args :?> Tensor<bool>) &&&& (Args.binaryY env.Args :?> Tensor<bool>) :> ITensor       


/// Logical Or.
type Or = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            (Args.binaryX env.Args :?> Tensor<bool>) |||| (Args.binaryY env.Args :?> Tensor<bool>) :> ITensor       


/// Logical Xor.
type Xor = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.bool [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = this.X.TypeName
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = 
            (Args.binaryX env.Args :?> Tensor<bool>) ^^^^ (Args.binaryY env.Args :?> Tensor<bool>) :> ITensor       


/// Equal.
type Equal = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).Equal (Args.binaryY env.Args)       


/// Not equal.
type NotEqual = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).NotEqual (Args.binaryY env.Args)       


/// Less than.
type Less = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).Less (Args.binaryY env.Args)       


/// Less then or equal.
type LessOrEqual = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).LessOrEqual (Args.binaryY env.Args)       


/// Greater than.
type Greater = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).Greater (Args.binaryY env.Args)       


/// Greater than or equal.
type GreaterOrEqual = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
        member this.Check () = Check.sameType [this.X; this.Y]; Check.sameShape [this.X; this.Y]
        member this.TypeName = TypeName.ofType<bool>
        member this.Shape = this.X.Shape
        member this.Args = Args.binary this.X this.Y
        member this.ReplaceArgs args = { this with X = Args.binaryX args; Y = Args.binaryY args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env = (Args.binaryX env.Args).GreaterOrEqual (Args.binaryY env.Args)       


/// Dot product.
type Dot = { X: BaseExpr; Y: BaseExpr } with
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
        member this.Eval env = (Args.binaryX env.Args).Dot (Args.binaryY env.Args)       


/// Tensor product.
type TensorProduct = { X: BaseExpr; Y: BaseExpr } with
    interface IOp with       
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
        member this.Eval env = (Args.binaryX env.Args).TensorProduct (Args.binaryY env.Args)       


/// Replace a slice of a tensor with another tensor.
type SetSubtensor = {X: BaseExpr; Y: BaseExpr; Range: SimpleRangesSpec} with
    interface IOp with      
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
                |> Map.map (fun _ v -> v :?> BaseExpr)
            Map.join xyArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
            let range = this.Range |> SimpleRangesSpec.replaceDynElems dynPrefix dynArgs               
            {this with X=Args.binaryX args; Y=Args.binaryY args; Range=range} :> _
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
            let trgt = Args.binaryX env.Args |> ITensor.copy
            trgt.[range] <- Args.binaryY env.Args
            trgt

