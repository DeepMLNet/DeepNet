namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor.Expr
open Tensor



/// Outputs each argument as a channel.
type Bundle = {ChExprs: Map<Ch, BaseExprCh>} with

    /// Argument corresponding to specifed channel.
    static member chToArg ch =
        match ch with
        | Ch.Default -> Arg.Only
        | Ch.Custom name -> Arg.Custom name
        | Ch.N n -> Arg.N n

    /// Channels corresponding to specifed argument.
    static member argToCh arg =
        match arg with
        | Arg.Only -> Ch.Default
        | Arg.Custom name -> Ch.Custom name
        | Arg.N n -> Ch.N n
        | _ -> failwithf "Argument %A not allowed for Bundle." arg

    interface IOp with
        member this.Check () = ()
        member this.Channels = 
            this.ChExprs |> Map.keys
        member this.TypeNames = 
            this.ChExprs |> Map.map (fun _ expr -> expr.TypeName)
        member this.Devs = 
            this.ChExprs |> Map.map (fun _ expr -> expr.Dev)
        member this.Shapes = 
            this.ChExprs |> Map.map (fun _ expr -> expr.Shape)
        member this.Args = 
            this.ChExprs |> Map.mapKeyValue (fun ch expr -> Bundle.chToArg ch, expr)
        member this.ReplaceArgs args = 
            {this with ChExprs=args |> Map.mapKeyValue (fun arg expr -> Bundle.argToCh arg, expr)} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            argVals |> Map.mapKeyValue (fun arg value -> Bundle.argToCh arg, value)
        


/// Discards the results of all arguments.
type Discard = {Xs: BaseExprCh list} with
    interface IOp with       
        member this.Check () = ()
        member this.Channels = Ch.onlyOne
        member this.TypeNames = TypeName.ofType<int32> |> Ch.only
        member this.Devs = HostTensor.Dev |> Ch.only
        member this.Shapes = ShapeSpec.emptyVector |> Ch.only
        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            HostTensor.zeros<int32> [0L] :> ITensor |> Ch.only



/// Build tensor using numeric ranges.
type BuildTensor = {Shape: ShapeSpec; Ranges: BaseRangesSpec list; Xs: BaseExprCh list} with
    interface IOp with       
        member this.Check () = 
            Check.sameType this.Xs
            Check.sameDev this.Xs
            if this.Ranges.Length <> this.Xs.Length then
                failwithf "BuildTensor ranges must match arguments, but got %d ranges and %d arguments."
                            this.Ranges.Length this.Xs.Length
            match ShapeSpec.tryEval this.Shape with
            | Some shp ->
                for rng, arg in List.zip this.Ranges this.Xs do
                    if rng.Length <> shp.Length then
                        failwithf "BuildTensor range %A has wrong dimensionality for shape %A." rng shp
                    for (start, stop), size, argSize in List.zip3 rng shp arg.Shape do
                        if argSize <> stop - start + 1L then
                            failwithf "BuildTensor range %A is invalid for argument of shape %A." rng arg.Shape
                        match Size.tryEval start, Size.tryEval stop with
                        | Some start, Some stop when not (0L <= start && start < size && 0L <= stop && 
                                                            stop < size && start <= stop) ->
                            failwithf "BuildTensor range %A is invalid for shape %A." rng shp
                        | _, _ -> ()
            | None -> ()       
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.Xs.Head.TypeName |> Ch.only
        member this.Devs = this.Xs.Head.Dev |> Ch.only
        member this.Shapes = this.Shape |> Ch.only
        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
        member this.SubstSymSizes env = 
            let sSize = Size.substSymbols env
            {this with Shape=ShapeSpec.substSymbols env this.Shape
                       Ranges=this.Ranges |> List.map (List.map (fun (f,l) -> sSize f, sSize l))} :> _
        member this.CanEvalAllSymSizes = 
            ShapeSpec.canEval this.Shape &&
            List.forall BaseRangesSpec.canEval this.Ranges
        member this.Eval env argVals = 
            let vs = ArgValue.naryXs argVals
            let trgt = vs.Head.ZerosOfSameType vs.Head.Dev (ShapeSpec.eval this.Shape)
            for rng, e in List.zip this.Ranges vs do                            
                let aryRng = rng |> List.map (fun (first, last) -> 
                    Rng.Rng (Some (Size.eval first), Some (Size.eval last)))
                trgt.[aryRng] <- e 
            trgt |> Ch.only



/// Elementwise calculated tensor.
type Elements = {Shape: ShapeSpec; ElemExpr: Elem.Expr; Xs: BaseExprCh list} with
    // TODO: introduce multi-channel element-wise calculation op.
    interface IOp with       
        member this.Check () = 
            Check.sameDev this.Xs
            let tns = this.Xs |> List.map (fun x -> x.TypeName)
            let ss = this.Xs |> List.map (fun x -> x.Shape)
            Elem.Expr.check this.ElemExpr |> ignore
            Elem.Expr.checkCompatibility this.ElemExpr ss tns this.Shape  
        member this.Channels = Ch.onlyOne            
        member this.TypeNames = Elem.Expr.typeName this.ElemExpr |> Ch.only
        member this.Devs = this.Xs.Head.Dev |> Ch.only
        member this.Shapes = this.Shape |> Ch.only
        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
        member this.SubstSymSizes env = 
            {this with Shape=ShapeSpec.substSymbols env this.Shape
                       ElemExpr=Elem.Expr.substSymSizes env this.ElemExpr} :> _
        member this.CanEvalAllSymSizes = 
            ShapeSpec.canEval this.Shape &&
            Elem.Expr.canEvalAllSymSizes this.ElemExpr
        member this.Eval env argVals = 
            let esv = ArgValue.naryXs argVals
            let nResShape = ShapeSpec.eval this.Shape
            Elem.Interpreter.evalUntyped this.ElemExpr esv nResShape 
            |> Ch.only



/// Elementwise interpolation using a value table.
type Interpolate = {Interpolator: Interpolator; Xs: BaseExprCh list} with
    interface IOp with       
        member this.Check () = 
            Check.sameType this.Xs
            Check.sameDev this.Xs
            let nDims = this.Interpolator.MinArg.Length
            if nDims < 1 then
                failwith "Interpolator must be at least one-dimensional."
            if this.Interpolator.MaxArg.Length <> nDims || this.Interpolator.Outside.Length <> nDims ||
                this.Interpolator.Resolution.Length <> nDims then
                    failwith "MinArg, MaxArg, Resolution and Outside have inconsistent lengths."
            if this.Xs.Length <> nDims then
                failwith "Number of arguments does not match dimensionality of interpolator."
            if not ((this.Interpolator.MinArg, this.Interpolator.MaxArg) 
                ||> List.forall2 (fun mi ma -> conv<float> mi < conv<float> ma)) then
                    failwith "MinArg of interpolator must be smaller than MaxArg."
            if this.Interpolator.Resolution |> List.exists ((>) 0.0) then
                failwith "Resolution of interpolator must be positive."
            for x in this.Xs do 
                if not (ShapeSpec.equalWithoutBroadcastability x.Shape this.Xs.Head.Shape) then
                    failwithf "All arguments to interpolator must have equal shape but got shapes %A and %A."
                                this.Xs.Head.Shape x.Shape
        member this.Channels = Ch.onlyOne
        member this.TypeNames = this.Xs.Head.TypeName |> Ch.only
        member this.Devs = this.Xs.Head.Dev |> Ch.only
        member this.Shapes = this.Xs.Head.Shape |> Ch.only
        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Eval env argVals = 
            let esv = ArgValue.naryXs argVals
            Interpolator.interpolateUntyped this.Interpolator esv |> Ch.only



