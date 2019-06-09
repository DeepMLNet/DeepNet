namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Base
open Tensor.Expr.Compiler
open Tensor
open Tensor.Backend


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillUnaryPlus (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillUnaryMinus (ArgValue.unaryX argVals))
        }


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
        
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillAbs (ArgValue.unaryX argVals))
        }

    
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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSgn (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillLog (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillLog10 (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillExp (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSin (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillCos (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillTan (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillAsin (ArgValue.unaryX argVals))
        }


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
        
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillAcos (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillAtan (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSinh (ArgValue.unaryX argVals))
        }


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
        
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillCosh (ArgValue.unaryX argVals))
        }


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
        
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillTanh (ArgValue.unaryX argVals))
        }
        

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
        
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSqrt (ArgValue.unaryX argVals))
        }


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
        
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillCeiling (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillFloor (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillRound (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillTruncate (ArgValue.unaryX argVals))
        }


/// (Batched) matrix inverse.
type Invert = { X: BaseExprCh } with
    interface IOp with      
        member this.Check () = 
            if this.X.NDims < 2 then
                failwithf "Need at least a matrix to invert but got shape %A" this.X.Shape
            if not (Size.equalIgnoringBc this.X.Shape.[this.X.NDims-2] this.X.Shape.[this.X.NDims-1]) then
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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                // TODO: So far temporary storage is allocated at runtime.
                //       This should be changed to also perform preallocations.
                (ChValue.onlyX chVals).FillInvert (ArgValue.unaryX argVals))
        }
                

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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.All)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals :?> Tensor<bool>).FillNegate (ArgValue.unaryX argVals :?> Tensor<bool>))
        }


/// Reshape
type Reshape = { X: BaseExprCh; Shape: Shape } with
    interface IOp with      
        member this.Check () = 
            if not (Size.equalIgnoringBc (Shape.nElem this.X.Shape) (Shape.nElem this.Shape)) then
                failwithf "Cannot change number of elements while reshaping from %A to %A." 
                            this.X.Shape this.Shape
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = 
            { this with Shape = Shape.subst env this.Shape } :> _
        member this.CanEvalAllSymSizes = 
            Shape.canEval this.Shape
        member this.Eval env argVals =
            (ArgValue.unaryX argVals) |> ITensor.reshape (Shape.eval this.Shape) |> Ch.only

    interface IStubWishingOp with
        member this.WishStubs data =
            CompileTools.propUnaryWish data (fun wish ->
                wish |> TensorStub.tryReshape (Shape.eval this.Shape))

    interface ICompilableOp with
        member this.Compile data =
            let shape = Shape.eval this.Shape
            let argShape = Shape.eval this.X.Shape
            let argStub = ArgValue.unaryX data.ArgStubs 

            let performCopy chVals argVals =
                let chVal = ChValue.onlyX chVals
                let argVal = ArgValue.unaryX argVals
                let trgtView = chVal |> ITensor.reshapeView argShape
                trgtView.CopyFrom argVal
            
            // Check if we made a wish for the argument stub.
            match data.ArgStubWishes |> Map.tryFind Arg.Only with
            | Some argStubWish when argStubWish = argStub ->
                // We propagated a tensor stub as a wish to our argument and it was accepted.
                // Thus our channel stub is already assigned and no actions need to be performed.
                {
                    ChStubs = data.ChStubs
                    Actions = CompileTools.noAction data
                }
            | Some _argStubWish ->
                // We propagated a tensor stub as a wish to our argument, but it was not accepeted.
                // Thus we need to copy the arguments output to our assigned channel stub.
                {
                    ChStubs = data.ChStubs
                    Actions = CompileTools.simpleAction data performCopy
                }
            | None ->
                // No wish was propagated, we have to compute a channel stub.
                if argStub.HasLayout then
                    // Layout is known, check if in-place reshape is possible.
                    match argStub |> TensorStub.tryReshape shape with
                    | Some chStub -> 
                        // Perform in-place reshape, return reshaped tensor.
                        {
                            ChStubs = Ch.only chStub
                            Actions = CompileTools.noAction data
                        }
                    | None -> 
                        // in-place reshape is impossible, allocate channel stub for copy
                        {
                            ChStubs = CompileTools.chStubs data
                            Actions = CompileTools.simpleAction data performCopy
                        }
                else
                    // Layout is unknown.
                    match argStub.Storage with
                    | StorageStub.Temporary _
                    | StorageStub.External _ ->
                        // Decide during execution, if copy must be performed.
                        {
                            ChStubs = Ch.only {
                                Shape = shape
                                TypeName = argStub.TypeName
                                Dev = argStub.Dev
                                OffsetStride = OffsetStride.Runtime (RuntimeStub ())
                                Storage = StorageStub.External (RuntimeStub ())
                            }
                            Actions = {new IAction with
                                member __.Execute execData =                            
                                    let argVal = execData.StubValue argStub
                                    let chVal = argVal |> ITensor.reshape shape
                                    {RuntimeChValues = Ch.only chVal}  
                                member __.Dev = argStub.Dev
                            }
                        }          
                    | StorageStub.Allocated _
                    | StorageStub.Fixed _ ->
                        // Allocate target storage and perform copy.
                        {
                            ChStubs = CompileTools.chStubs data
                            Actions = CompileTools.simpleAction data performCopy
                        }

    interface IOpFormat with
        member this.Text =
            sprintf "Reshape%A" this.Shape


/// Broadcast.
type DoBroadcast = { X: BaseExprCh; Shape: Shape } with
    interface IOp with      
        member this.Check () = 
            if Shape.nDim this.X.Shape <> Shape.nDim this.Shape then
                failwithf "Tensor of shape %A does not have same number of dimesions as broadcast shape %A."
                            this.X.Shape this.Shape
            for dim in 0 .. (Shape.nDim this.Shape) - 1 do
                match this.X.Shape.[dim], this.Shape.[dim] with
                | Size.Broadcast, _ -> ()
                | ssa, ssb when not (Size.equalIgnoringBc ssa ssb) -> 
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
            { this with Shape = Shape.subst env this.Shape } :> _
        member this.CanEvalAllSymSizes = 
            Shape.canEval this.Shape
        member this.Eval env argVals = 
            (ArgValue.unaryX argVals) |> ITensor.broadcastTo (Shape.eval this.Shape) |> Ch.only

    interface ICompilableOp with
        member this.Compile data =
            let shape = Shape.eval this.Shape
            CompileTools.tryStatic data
                (TensorStub.tryBroadcastTo shape)
                (ITensor.broadcastTo shape)

    interface IOpFormat with
        member this.Text =
            sprintf "DoBroadcast%A" this.Shape


/// Permute the axes.
type PermuteAxes = {X: BaseExprCh; Permutation: int list} with
    interface IOp with      
        member this.Check () = 
            if Shape.nDim this.X.Shape <> List.length this.Permutation then
                failwithf "Permutation %A must have same rank as shape %A." this.Permutation this.X.Shape
            if not (Permutation.is this.Permutation) then
                failwithf "%A is not a valid permutation of an %d-dimensional tensor." 
                            this.Permutation (Shape.nDim this.X.Shape)
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.permuteAxes this.Permutation |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals) |> ITensor.permuteAxes this.Permutation |> Ch.only

    interface IStubWishingOp with
        member this.WishStubs data =
            CompileTools.propUnaryWish data (fun wish ->
                wish |> TensorStub.tryPermuteAxes (Permutation.invert this.Permutation))

    interface ICompilableOp with
        member this.Compile data =
            CompileTools.tryStatic data
                (TensorStub.tryPermuteAxes this.Permutation)
                (ITensor.permuteAxes this.Permutation)

    interface IOpFormat with
        member this.Text =
            sprintf "PermuteAxes%A" this.Permutation


/// Read a slice from a tensor.
type Subtensor = {X: BaseExprCh; Range: SimpleRanges} with
    
    static member internal evalRange (range: SimpleRanges) (argVals: Map<Arg, ITensor>) =
        let dynVals = 
            argVals 
            |> Map.filter (fun arg _ -> 
                match arg with
                | Arg.N _ -> true
                | _ -> false)
            |> Map.map (fun _ v -> Tensor.value (v :?> Tensor<int64>) |> Size.fix)         
        range 
        |> SimpleRangesArgs.resolveDynElems dynVals 
        |> SimpleRanges.eval   

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
                | SimpleRange.SymStartSymEnd (s, fo)    -> (fo |? (shp - Size.one)) + 1L - s
                | SimpleRange.DynStartSymSize (_, size) -> size)            
            |> Ch.only
        member this.Args = 
            let xArgs = Args.unary this.X 
            let dynArgs = 
                SimpleRangesArgs.toArgs this.Range
                |> Map.map (fun _ v -> v :?> BaseExprCh)
            Map.join xArgs dynArgs
        member this.ReplaceArgs args = 
            let dynArgs = args |> Map.map (fun _ v -> v :> IDynElem)
            let range = this.Range |> SimpleRangesArgs.replaceFromArgs dynArgs               
            {this with X=Args.unaryX args; Range=range} :> _
        member this.SubstSymSizes env = {this with Range = SimpleRanges.subst env this.Range} :> _
        member this.CanEvalAllSymSizes = SimpleRanges.canEvalSymbols this.Range
        member this.Eval env argVals = 
            let range = Subtensor.evalRange this.Range argVals
            (ArgValue.unaryX argVals).[range] |> Ch.only

    interface ICompilableOp with
        member this.Compile data =
            let srcStub = data.ArgStubs.[Arg.Only]
            let dynamicChStub = Ch.only {
                Shape = (this :> IOp).Shapes.[Ch.Default] |> Shape.eval
                TypeName = (this :> IOp).TypeNames.[Ch.Default]
                Dev = (this :> IOp).Devs.[Ch.Default]
                OffsetStride = OffsetStride.Runtime (RuntimeStub ())
                Storage = data.ArgStubs.[Arg.Only].Storage                       
            }

            if SimpleRanges.isDynamic this.Range then
                // Dynamic range gives a dynamic tensor stub.
                // If the range arguments are stored on another device, they
                // are copied to host to calculate the offset and strides of the result.
                // An alternative would be to copy the tensor range to avoid the
                // transfer of the indices to the host.
                {
                    ChStubs = dynamicChStub
                    Actions = {new IAction with
                        member __.Execute execData =
                            let srcVal = execData.StubValue srcStub
                            let argValues = data.ArgStubs |> Map.map (fun _ stub -> execData.StubValue stub)
                            let range = Subtensor.evalRange this.Range argValues
                            let chVal = srcVal.[range] 
                            {RuntimeChValues = Ch.only chVal}
                        member __.Dev = this.X.Dev 
                    }
                }
            else
                // Static range allows to slice argument during compilation, when
                // its layout is known.
                let range = SimpleRanges.eval this.Range 
                match srcStub |> TensorStub.tryView range with
                | Some viewStub -> 
                    // Source layout is known and thus we can directly calculate the view.
                    {
                        ChStubs = Ch.only viewStub
                        Actions = CompileTools.noAction data
                    }
                | None ->
                    // Source layout is unknown and thus we have to calculate view during execution.
                    {
                        ChStubs = dynamicChStub
                        Actions = {new IAction with
                            member __.Execute execData =
                                let srcVal = execData.StubValue srcStub
                                let chVal = srcVal.[range] 
                                {RuntimeChValues = Ch.only chVal}
                            member __.Dev = this.X.Dev
                        }
                    }            

    interface IOpFormat with
        member this.Text =
            sprintf "Subtensor%A" this.Range


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

    interface IStubWishingOp with
        member this.WishStubs data =
            CompileTools.propUnaryWish data (fun wish ->
                wish |> TensorStub.tryReverseAxis this.Axis)

    interface ICompilableOp with
        member this.Compile data =
            CompileTools.tryStatic data 
                (TensorStub.tryReverseAxis this.Axis) 
                (ITensor.reverseAxis this.Axis)

    interface IOpFormat with
        member this.Text =
            sprintf "ReverseAxis<%d>" this.Axis


/// Extract the diagonal(s) along the given axes.
type Diag = {X: BaseExprCh; Axis1: int; Axis2: int} with
    interface IOp with      
        member this.Check () = 
            Check.axis this.Axis1 this.X
            Check.axis this.Axis2 this.X 
            if not (this.Axis1 < this.Axis2) then 
                failwith "First axis for extracting diagonal must come before second axis."
            if not (Size.equalIgnoringBc this.X.Shape.[this.Axis1] this.X.Shape.[this.Axis2]) then
                failwithf "Cannot extract diagonal along axes %d and %d from non-square tensor with shape %A" 
                            this.Axis1 this.Axis2 this.X.Shape
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis2 |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = {this with X = Args.unaryX args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).DiagAxis this.Axis1 this.Axis2 |> Ch.only

    interface ICompilableOp with
        member this.Compile data =
            CompileTools.tryStatic data 
                (TensorStub.tryDiagAxis this.Axis1 this.Axis2) 
                (fun arg -> arg.DiagAxis this.Axis1 this.Axis2)

    interface IOpFormat with
        member this.Text =
            sprintf "Diag<%d,%d>" this.Axis1 this.Axis2


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

    // TODO: wishing and wish propagation
    // For now we always copy the diagonal into a zero-filled tensor.
    // This should be avoidable when our argument accepts our stub wish.

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                let res = ChValue.onlyX chVals
                res.FillZero ()
                let resDiagView = res.DiagAxis this.Axis1 this.Axis2 
                resDiagView.CopyFrom (ArgValue.unaryX argVals))
        }

    interface IOpFormat with
        member this.Text =
            sprintf "DiagMat<%d,%d>" this.Axis1 this.Axis2


/// Sum over specified axis.
type SumAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).SumAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillSumAxis this.Axis (ArgValue.unaryX argVals))
        }


/// Product over specified axis.
type ProductAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).ProductAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillProductAxis this.Axis (ArgValue.unaryX argVals))
        }


/// Maximum over specified axis.
type MaxAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).MaxAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillMaxAxis this.Axis (ArgValue.unaryX argVals))
        }


/// Minimum over specified axis.
type MinAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).MinAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillMinAxis this.Axis (ArgValue.unaryX argVals))
        }


/// Maximum over specified axis.
type ArgMaxAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int64> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).ArgMaxAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillArgMaxAxis this.Axis (ArgValue.unaryX argVals))
        }


/// Minimum over specified axis.
type ArgMinAxis = {X: BaseExprCh; Axis: int} with
    interface IOp with      
        member this.Check () = Check.axis this.Axis this.X
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int64> |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Shape.withoutAxis this.Axis |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = (ArgValue.unaryX argVals).ArgMinAxis this.Axis |> Ch.only

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillArgMinAxis this.Axis (ArgValue.unaryX argVals))
        }


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
                | None when dim >= Shape.nDim trgtShape ->
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
            (ArgValue.unaryX argVals).Gather vIndices |> Ch.only

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                let vIndices = argVals |> ArgValue.naryOptXs this.Indices.Length
                (ChValue.onlyX chVals).FillGather vIndices (ArgValue.unaryX argVals))
        }


/// Disperses elements according to the specified index tensors.
type Scatter = {X: BaseExprCh; Indices: BaseExprCh option list; Shape: Shape} with
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
            (ArgValue.unaryX argVals).Scatter vIndices (Shape.eval this.Shape) |> Ch.only

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                let vIndices = argVals |> ArgValue.naryOptXs this.Indices.Length
                (ChValue.onlyX chVals).FillScatter vIndices (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.passthroughStub data
            Actions = CompileTools.noAction data
        }
    

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
    
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.passthroughStub data
            Actions = CompileTools.noAction data
        }


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
        
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.passthroughStub data
            Actions = CompileTools.noAction data
        }

    
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
            Ch.only v
            
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.passthroughStub data
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                let v = ArgValue.unaryX argVals
                printfn "%s=\n%A\n" this.Label v)
        }


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
            Ch.only v
            
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.passthroughStub data
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                let v = ArgValue.unaryX argVals
                Dump.dumpValue this.Dataset v)
        }


/// A non-finite value was encountered.
exception NonFiniteValueException of msg:string with 
    /// <summary>Detailed error message.</summary>    
    override __.Message = __.msg


/// If the value contains NaNs or infinities, outputs their location and 
/// stops the computation.
type CheckFinite = {Label: string; X: BaseExprCh} with
    member internal this.DoCheck (v: ITensor) =
        if not (v.AllFinite ()) then
            let msg = 
                sprintf "Non-finite value encountered in %s with value:\n%A" this.Label v
            raise (NonFiniteValueException msg)

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
            this.DoCheck v
            v |> Ch.only                    
            
    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.passthroughStub data
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                let v = ArgValue.unaryX argVals
                this.DoCheck v)
        }


/// Converts the data to the specified type.
type Convert = {ToType: TypeName; X: BaseExprCh} with
    interface IOp with
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.ToType |> Ch.only
        member this.Devs = this.X.Dev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            v |> ITensor.convertToType this.ToType.Type |> Ch.only     

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                (ChValue.onlyX chVals).FillConvert (ArgValue.unaryX argVals))
        }


/// Transfers the data to the specified device.
type Transfer = {ToDev: ITensorDevice; X: BaseExprCh} with
    interface IOp with
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.X.TypeName |> Ch.only
        member this.Devs = this.ToDev |> Ch.only
        member this.Shapes = this.X.Shape |> Ch.only
        member this.Args = Args.unary this.X
        member this.ReplaceArgs args = { this with X = Args.unaryX args } :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let v = ArgValue.unaryX argVals
            v |> ITensor.transfer this.ToDev |> Ch.only     

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.chStubs (data, tryInplace=TryInplace.None)
            Actions = CompileTools.simpleAction data (fun chVals argVals ->
                // TODO: Do we need to take care about cross-device synchronization here? 
                (ChValue.onlyX chVals).TransferFrom (ArgValue.unaryX argVals))
        }


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

    interface ICompilableOp with
        member this.Compile data = {
            ChStubs = CompileTools.passthroughStub data
            Actions = CompileTools.noAction data
        }


