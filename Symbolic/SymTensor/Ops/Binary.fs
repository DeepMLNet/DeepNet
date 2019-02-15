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
        member this.Eval env = (ArgValue.binaryX env.Args).Add (ArgValue.binaryY env.Args)      


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
        member this.Eval env = (ArgValue.binaryX env.Args).Subtract (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).Multiply (ArgValue.binaryY env.Args)      


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
        member this.Eval env = (ArgValue.binaryX env.Args).Divide (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).Pow (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).Modulo (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).MaxElemwise (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).MinElemwise (ArgValue.binaryY env.Args)       


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
            Map ["Cond", Arg.Expr this.Cond
                 "IfTrue", Arg.Expr this.IfTrue
                 "IfFalse", Arg.Expr this.IfFalse]
        member this.ReplaceArgs args = 
            {this with Cond=Arg.expr args.["Cond"]
                       IfTrue=Arg.expr args.["IfTrue"]
                       IfFalse=Arg.expr args.["IfFalse"]} :> _
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
            (ArgValue.binaryX env.Args :?> Tensor<bool>) &&&& (ArgValue.binaryY env.Args :?> Tensor<bool>) :> ITensor       


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
            (ArgValue.binaryX env.Args :?> Tensor<bool>) |||| (ArgValue.binaryY env.Args :?> Tensor<bool>) :> ITensor       


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
            (ArgValue.binaryX env.Args :?> Tensor<bool>) ^^^^ (ArgValue.binaryY env.Args :?> Tensor<bool>) :> ITensor       


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
        member this.Eval env = (ArgValue.binaryX env.Args).Equal (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).NotEqual (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).Less (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).LessOrEqual (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).Greater (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).GreaterOrEqual (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).Dot (ArgValue.binaryY env.Args)       


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
        member this.Eval env = (ArgValue.binaryX env.Args).TensorProduct (ArgValue.binaryY env.Args)       


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
                |> Map.map (fun _ v -> v :?> BaseExpr |> Arg.Expr)
            Map.join xyArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v |> Arg.expr :> IDynElem)
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
            let trgt = ArgValue.binaryX env.Args |> ITensor.copy
            trgt.[range] <- ArgValue.binaryY env.Args
            trgt

