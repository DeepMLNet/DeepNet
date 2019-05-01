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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillAdd (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSubtract (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillMultiply (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillDivide (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillPow (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillModulo (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillMaxElemwise (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillMinElemwise (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


/// Element-wise if-then-else.
type IfThenElse = {Cond: BaseExprCh; IfTrue: BaseExprCh; IfFalse: BaseExprCh} with
    static member private argCond = Arg.Custom "Cond"
    static member private argIfTrue = Arg.Custom "IfTrue"
    static member private argIfFalse = Arg.Custom "IfFalse"

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
            Map [IfThenElse.argCond, this.Cond
                 IfThenElse.argIfTrue, this.IfTrue
                 IfThenElse.argIfFalse, this.IfFalse]
        member this.ReplaceArgs args = 
            {this with Cond=args.[IfThenElse.argCond]
                       IfTrue=args.[IfThenElse.argIfTrue]
                       IfFalse=args.[IfThenElse.argIfFalse]} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            argVals.[IfThenElse.argIfTrue].IfThenElse 
                argVals.[IfThenElse.argIfFalse] argVals.[IfThenElse.argCond]
            |> Ch.only

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillIfThenElse 
                    argVals.[IfThenElse.argCond] 
                    argVals.[IfThenElse.argIfTrue] 
                    argVals.[IfThenElse.argIfFalse])
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals :?> Tensor<bool>).FillAnd 
                    (ArgValue.binaryX argVals :?> Tensor<bool>) 
                    (ArgValue.binaryY argVals :?> Tensor<bool>))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals :?> Tensor<bool>).FillOr
                    (ArgValue.binaryX argVals :?> Tensor<bool>) 
                    (ArgValue.binaryY argVals :?> Tensor<bool>))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals :?> Tensor<bool>).FillXor
                    (ArgValue.binaryX argVals :?> Tensor<bool>) 
                    (ArgValue.binaryY argVals :?> Tensor<bool>))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillEqual (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillNotEqual (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillLess (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillLessOrEqual (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillGreater (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillGreaterOrEqual (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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
        
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillDot (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillTensorProduct (ArgValue.binaryX argVals) (ArgValue.binaryY argVals))
        }


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
            let range = Subtensor.evalRange this.Range argVals
            let trgt = ArgValue.binaryX argVals |> ITensor.copy
            trgt.[range] <- ArgValue.binaryY argVals
            trgt |> Ch.only

    // No wishing or wish propagation of argument stubs is possible at the moment,
    // since our argument may be shared and we thus cannot overwrite parts of
    // its contents.

    interface ICompilableOp with
        member this.Compile data = 
            // Try to use first argument in-place and overwrite the part of
            // the tensor we have to set.
            let chStubs = CompileTools.chStubs (data, tryInplace=TryInplace.Limited (Set [Arg.X]))

            let copyActions = 
                if data.ArgStubs.[Arg.X] = data.ChStubs.[Ch.Default] then
                    // We are operation in-place the first argument.
                    // Thus it does not need to be copied.
                    CompileTools.noAction data
                else 
                    // We are not operating in-place first argument.
                    // Thus we need to copy it into result tensor.
                    CompileTools.simpleAction data (fun chVals argVals ->
                        (ChValue.onlyX chVals).CopyFrom (ArgValue.binaryX argVals))

            let setActions = 
                if SimpleRanges.isDynamic this.Range then
                    CompileTools.simpleAction data (fun chVals argVals ->
                        let range = Subtensor.evalRange this.Range argVals
                        (ChValue.onlyX chVals).[range] <- ArgValue.binaryY argVals)
                else
                    let range = SimpleRanges.eval this.Range 
                    // TODO: The range tensor stub could be pre-computed at compile-time,
                    //       but at the moment there is no method to lookup a tensor
                    //       corresponding to a stub at execution time.
                    CompileTools.simpleAction data (fun chVals argVals ->
                        (ChValue.onlyX chVals).[range] <- ArgValue.binaryY argVals)                   
        
            {
                ChStubs = chStubs
                Actions = CompileTools.concatActions [copyActions; setActions]
            }
       
    interface IOpFormat with
        member this.Text =
            sprintf "SetSubtensor%A" this.Range
