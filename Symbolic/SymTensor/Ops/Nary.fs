namespace SymTensor.Ops

open SymTensor


/// Discards the results of all arguments.
type Discard = {Xs: Expr2 list} with
    interface IOp2 with       
        member this.Check () = ()
        member this.TypeName = TypeName.ofType<int32>
        member this.Shape = ShapeSpec.emptyVector
        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp // TODO
        member this.Eval env = HostTensor.zeros<int32> [0L] :> ITensor
let (|Discard|_|) (expr: Expr2) =
    match expr.Op with
    | :? Discard as this -> Some (this.Xs)
    | _ -> None

/// Discards the results of all arguments.
let discard (xs: Expr2 list) =
    {Discard.Xs=xs} |> Expr2

/// Build tensor using numeric ranges.
type BuildTensor = {Shape: ShapeSpec; Ranges: BaseRangesSpec list; Xs: Expr2 list} with
    interface IOp2 with       
        member this.Check () = 
            Check.sameType this.Xs
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
                        match SizeSpec.tryEval start, SizeSpec.tryEval stop with
                        | Some start, Some stop when not (0L <= start && start < size && 0L <= stop && 
                                                            stop < size && start <= stop) ->
                            failwithf "BuildTensor range %A is invalid for shape %A." rng shp
                        | _, _ -> ()
            | None -> ()       
        member this.TypeName = this.Xs.Head.TypeName
        member this.Shape = this.Shape
        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
        member this.SubstSymSizes env = 
            let sSize = SizeSpec.substSymbols env
            {this with Shape=ShapeSpec.substSymbols env this.Shape
                        Ranges=this.Ranges |> List.map (List.map (fun (f,l) -> sSize f, sSize l))} :> _
        member this.CanEvalAllSymSizes = 
            ShapeSpec.canEval this.Shape &&
            List.forall BaseRangesSpec.canEval this.Ranges
        member this.Deriv dOp = Args.binary dOp dOp // TODO
        member this.Eval env = 
            let vs = Args.naryXs env.Args
            let trgt = vs.Head.ZerosOfSameType vs.Head.Dev (ShapeSpec.eval this.Shape)
            for rng, e in List.zip this.Ranges vs do                            
                let aryRng = rng |> List.map (fun (first, last) -> 
                    Rng.Rng (Some (SizeSpec.eval first), Some (SizeSpec.eval last)))
                trgt.[aryRng] <- e 
            trgt
let (|BuildTensor|_|) (expr: Expr2) =
    match expr.Op with
    | :? BuildTensor as this -> Some this
    | _ -> None

/// Build tensor from numeric ranges.
let internal buildTensor shape ranges xs =
    {BuildTensor.Shape=shape; Ranges=ranges; Xs=xs} |> Expr2
    
/// Elementwise calculated tensor.
type Elements = {Shape: ShapeSpec; ElemExpr: Elem.Expr; Xs: Expr2 list} with
    interface IOp2 with       
        member this.Check () = 
            let tns = this.Xs |> List.map Expr2.typeName
            let ss = this.Xs |> List.map Expr2.shape
            Elem.Expr.check this.ElemExpr |> ignore
            Elem.Expr.checkCompatibility this.ElemExpr ss tns this.Shape   
        member this.TypeName = Elem.Expr.typeName this.ElemExpr
        member this.Shape = this.Shape
        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
        member this.SubstSymSizes env = 
            let sSize = SizeSpec.substSymbols env
            {this with Shape=ShapeSpec.substSymbols env this.Shape
                        ElemExpr=Elem.Expr.substSymSizes env this.ElemExpr} :> _
        member this.CanEvalAllSymSizes = 
            ShapeSpec.canEval this.Shape &&
            Elem.Expr.canEvalAllSymSizes this.ElemExpr
        member this.Deriv dOp = Args.binary dOp dOp // TODO
        member this.Eval env = 
            let esv = Args.naryXs env.Args
            let nResShape = ShapeSpec.eval this.Shape
            Elem.Interpreter.evalUntyped this.ElemExpr esv nResShape 
let (|Elements|_|) (expr: Expr2) =
    match expr.Op with
    | :? Elements as this -> Some this
    | _ -> None

/// Calculates a tensor elementwise using the given element expression and result shape.
let elements shape elemExpr xs =
    {Elements.Shape=shape; ElemExpr=elemExpr; Xs=xs} |> Expr2

/// Elementwise interpolation using a value table.
type Interpolate = {Interpolator: Interpolator; Xs: Expr2 list} with
    interface IOp2 with       
        member this.Check () = 
            Check.sameType this.Xs
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
        member this.TypeName = this.Xs.Head.TypeName
        member this.Shape = this.Xs.Head.Shape
        member this.Args = Args.nary this.Xs
        member this.ReplaceArgs args = {this with Xs=Args.naryXs args} :> _
        member this.SubstSymSizes env = this :> _
        member this.CanEvalAllSymSizes = true
        member this.Deriv dOp = Args.binary dOp dOp // TODO
        member this.Eval env = 
            let esv = Args.naryXs env.Args
            Interpolator.interpolateUntyped this.Interpolator esv
let (|Interpolate|_|) (expr: Expr2) =
    match expr.Op with
    | :? Interpolate as this -> Some this
    | _ -> None

/// Element-wise n-dimensional interpolation using the specified interpolator.
/// The interpolator is created using the Interpolator.create function.
let interpolate interpolator xs =
    let xs = Expr2.broadcastToSameMany xs
    {Interpolate.Interpolator=interpolator; Xs=xs} |> Expr2

/// Element-wise one-dimensional interpolation using the specified interpolator.
/// The interpolator is created using the Interpolator.create function.
let interpolate1D interpolator x =
    interpolate interpolator [x]

/// Element-wise two-dimensional interpolation using the specified interpolator.
/// The interpolator is created using the Interpolator.create function.
let interpolate2D interpolator x y =
    interpolate interpolator [x; y]

/// Element-wise three-dimensional interpolation using the specified interpolator.
/// The interpolator is created using the Interpolator.create function.
let interpolate3D interpolator x y z =
    interpolate interpolator [x; y; z]

