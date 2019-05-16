namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor
open Tensor.Expr



[<RequireQualifiedAccess>]
type TryInplace =
    | All
    | Limited of Set<Arg>
    | None


type CompileTools () =

    /// Allocates tensor stubs for all output channels of the op.
    /// Argument stubs will be reused, if tryInplace is true.
    /// Channel wishes will be honored, if honorWishes is true.
    static member chStubs (data: CompileData, ?tryInplace: TryInplace, ?honorWishes: bool) =
        let tryInplace = defaultArg tryInplace TryInplace.None
        let honorWishes = defaultArg honorWishes true

        let op = data.Expr.Op
        let mutable overwritableArgs = data.OverwritableArgs
        
        /// Returns an overwritable argument TensorStub that matches the
        /// given specifications, if available.
        let tryUseArg typeName dev shape =
            overwritableArgs
            |> Set.toSeq
            |> Seq.filter (fun arg ->
                // check that argument may be overridden
                match tryInplace with
                | TryInplace.All -> true
                | TryInplace.Limited allowed when allowed.Contains arg -> true
                | TryInplace.Limited _allowed -> false
                | TryInplace.None -> false)
            |> Seq.tryFind (fun arg ->
                // match type, device and shape
                let argExpr = op.Args.[arg]
                argExpr.TypeName = typeName &&
                argExpr.Dev = dev &&
                Shape.eval argExpr.Shape = shape)
            |> Option.map (fun arg -> 
                // remove from set of overwritable arguments
                overwritableArgs <- overwritableArgs |> Set.remove arg
                data.ArgStubs.[arg])


        op.Channels
        |> Set.toSeq
        |> Seq.map (fun ch ->
            match data.ChStubs |> Map.tryFind ch with
            | Some stub -> 
                // Channel stub has been pre-assigned due to propagated wish.
                ch, stub
            | None ->
                // We are free to choose the channels.
                let typeName = op.TypeNames.[ch]
                let shape = Shape.eval op.Shapes.[ch]
                let dev = op.Devs.[ch]
                let stub = 
                    match data.ChStubWishes |> Map.tryFind ch with
                    | Some chStubWish when honorWishes -> chStubWish
                    | _ ->
                        match tryUseArg typeName dev shape with
                        | Some inplaceStub -> inplaceStub
                        | None -> TensorStub.alloc (data.Alloc, typeName, shape, dev)
                ch, stub)
        |> Map.ofSeq

    /// Passes through tensor stub of unary argument.
    static member passthroughStub (data: CompileData) =
        Map [Ch.Default, data.ArgStubs.[Arg.Only]]

    /// Propagates a tensor stub wish for an unary operation.
    static member propUnaryWish (data: UpPropData) (fn: TensorStub -> TensorStub option)  =
        let chWishOpt = data.ChStubWishes |> Map.tryFind Ch.Default
        let argWishOpt = chWishOpt |> Option.bind fn
        match chWishOpt, argWishOpt with
        | Some chWish, Some argWish -> 
            // We had a wish for our channel stub and propagated it to our argument.
            // Thus we must commit to the wish and assign it to our channel.
            {
                ChStubs = Map [Ch.Default, chWish]
                ArgStubWishes = Map [Arg.Only, argWish]
            }
        | _ -> 
            // We had not wish or did not propagate it.
            {
                ChStubs = Map.empty
                ArgStubWishes = Map.empty
            }


    static member simpleAction (data: CompileData) (actFn: Map<Ch, ITensor> -> Map<Arg, ITensor> -> unit) =
        let dev = data.ChStubs.[Ch.Default].Dev
        {new IAction with
            member __.Execute execData =
                let chValues = data.ChStubs |> Map.map (fun _ stub -> execData.StubValue stub) 
                let argValues = data.ArgStubs |> Map.map (fun _ stub -> execData.StubValue stub)
                actFn chValues argValues 
                {RuntimeChValues=Map.empty}
            member __.Dev = dev
        }


    static member noAction (data: CompileData) =
        CompileTools.simpleAction data (fun _ _ -> ())


    static member concatActions (actions: IAction list) =
        {new IAction with
            member __.Execute execData =
                for act in actions.[0 .. actions.Length-2] do
                    act.Execute execData |> ignore
                (List.last actions).Execute execData
            member __.Dev = (List.last actions).Dev
        }


    /// Tries to apply `staticFn`, which only changes the layout, to the tensor stub of the argument.
    /// If this succeeds, no actions are performed at run-time.
    /// If it fails, `dynFn`, which also only changes the layout, is applied at run-time.
    static member tryStatic (data: CompileData) (staticFn: TensorStub -> TensorStub option) (dynFn: ITensor -> ITensor) =
        let op = data.Expr.Op
        let argStub = ArgValue.unaryX data.ArgStubs 

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
                Actions = CompileTools.simpleAction data (fun chVals argVals ->
                    (ChValue.onlyX chVals).CopyFrom (ArgValue.unaryX argVals))
            }
        | None ->
            // No wish was propagated, we have to compute a channel stub.
            match staticFn argStub with
            | Some chStub -> 
                // Channel stub could be calculated at compile-time.
                assert (chStub.Storage = argStub.Storage)
                {
                    ChStubs = Ch.only chStub
                    Actions = CompileTools.noAction data
                }
            | None ->
                // Channel stub must be calculated at run-time.
                {
                    ChStubs = Ch.only {
                        Shape = Shape.eval op.Shapes.[Ch.Default]
                        TypeName = op.TypeNames.[Ch.Default]
                        Dev = op.Devs.[Ch.Default]
                        OffsetStride = OffsetStride.Runtime (RuntimeStub ())
                        Storage = argStub.Storage    
                    }
                    Actions = 
                        {new IAction with
                            member __.Execute execData =
                                let argVal = execData.StubValue argStub
                                let chVal = dynFn argVal 
                                assert (chVal.Storage = argVal.Storage)
                                {RuntimeChValues = Map [Ch.Default, chVal]}
                            member __.Dev =
                                op.Devs.[Ch.Default]
                        }
                }




