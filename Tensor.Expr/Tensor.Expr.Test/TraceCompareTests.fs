namespace global
open System

#nowarn "25"

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Expr.Ops
open TestUtils


type TraceCompareTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format

    let realTypes = [typeof<single>; typeof<double>]

    [<TraceCompareFact>]
    let ``matrix-matrix dot`` () =   
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[6L; 3L]; [3L; 2L]] 
            (fun [a; b] ->  a .* b)

    [<TraceCompareFact>]
    let ``matrix-vector dot`` () =   
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[6L; 3L]; [3L]] 
            (fun [a; b] -> a .* b)

    [<TraceCompareFact>]
    let ``batched matrix-matrix dot`` () =   
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 5L; 6L; 3L]; [7L; 5L; 3L; 2L]] 
            (fun [a; b] -> a .* b)

    [<TraceCompareFact>]    
    let ``batched matrix-matrix dot with broadcasting`` () =   
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 5L; 6L; 3L]; [7L; -1L; 3L; 2L]] 
            (fun [a; b] -> a .* b)

    [<TraceCompareFact>]    
    let ``batched build diagonal`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 5L; 3L]] 
            (fun [a] -> UExpr.diagMat a)

    [<TraceCompareFact>]    
    let ``batched extract diagonal`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 5L; 4L; 4L]] 
            (fun [a] -> UExpr.diag a)

    [<TraceCompareFact>]    
    let ``matrix inverse`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[3L; 3L]] 
            (fun [a] -> UExpr.invert a)

    [<TraceCompareFact>]    
    let ``transposed matrix inverse`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[5L; 5L]] 
            (fun [a] -> UExpr.invert a.T)

    [<TraceCompareFact>]  
    let ``batched matrix inverse`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 3L; 4L; 4L]] 
            (fun [a] -> UExpr.invert a)

    [<TraceCompareFact>] 
    let ``sum`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 3L; 4L; 5L]] 
            (fun [a] -> UExpr.sum a)

    [<TraceCompareFact>]
    let ``sum axis 1`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 3L; 4L; 5L]] 
            (fun [a] -> UExpr.sumAxis 1 a)

    [<TraceCompareFact>]
    let ``sum axis 2`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 3L; 4L; 5L]] 
            (fun [a] -> a |> UExpr.sumAxis 3 |> UExpr.sumAxis 0)

    [<TraceCompareFact>]    
    let ``large sum axis`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 200L]] 
            (fun [a] -> a |> UExpr.sumAxis 0)

    [<TraceCompareFact>]    
    let ``product`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output [typeof<double>]
            [[7L; 3L; 4L; 5L]] 
            (fun [a] -> UExpr.product a)

    [<TraceCompareFact>]
    let ``product axis 1`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[7L; 3L; 4L; 5L]] 
            (fun [a] -> UExpr.productAxis 1 a)

    [<TraceCompareFact>]
    let ``max, min elemwise`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[3L; 3L]; [3L; 3L]; [3L; 3L]] 
            (fun [a; b; c]  -> UExpr.minElemwise (UExpr.maxElemwise a b) c)

    [<TraceCompareFact>]    
    let ``max, min elemwise derivative`` () =
        requireEqualTracesWithRandomDataAndTypesMultiChannel 
            output realTypes
            [[2L; 2L]; [2L; 2L]; [2L; 2L]] 
            (fun [a; b; c]  ->
                let expr = UExpr.minElemwise (UExpr.maxElemwise a b) c
                let dexpr = Deriv.compute expr
                Map [
                    Ch.Default, expr
                    Ch.Custom "da", dexpr.Wrt (extractVar a)
                    Ch.Custom "db", dexpr.Wrt (extractVar b)
                    Ch.Custom "dc", dexpr.Wrt (extractVar c)
                ] |> MultiChannelExpr.bundle)

    [<TraceCompareFact>]   
    let ``max, min reduction`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[4L; 5L; 3L]] 
            (fun [a]  -> a |> UExpr.maxAxis 2 |> UExpr.minAxis 1)

    [<TraceCompareFact>]    
    let ``max, min reduction derivative`` () =
        requireEqualTracesWithRandomDataAndTypesMultiChannel 
            output realTypes
            [[4L; 5L; 3L]] 
            (fun [a]  ->
                let expr = a |> UExpr.maxAxis 2 |> UExpr.minAxis 1
                let dexpr = Deriv.compute expr
                Map [
                    Ch.Default, expr
                    Ch.Custom "da", dexpr.Wrt (extractVar a)
                ] |> MultiChannelExpr.bundle)

    [<TraceCompareFact>]
    let ``argMax reduction`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[4L; 5L; 3L]] 
            (fun [a]  -> a |> UExpr.argMaxAxis 1)

    [<TraceCompareFact>]  
    let ``argMin reduction`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[4L; 5L; 3L]] 
            (fun [a]  -> a |> UExpr.argMinAxis 2)

    [<TraceCompareFact>]   
    let ``comparison`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[3L; 3L]; [3L; 3L]] 
            (fun [a; b] -> a >>== b)

    [<TraceCompareFact>]  
    let ``comparison, logics`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[3L; 3L]; [3L; 3L]; [3L; 3L]] 
            (fun [a; b; c] -> a >>== b &&&& ~~~~(b <<<< c))

    [<TraceCompareFact>]    
    let ``comparison, logics, conditionals`` () =
        requireEqualTracesWithRandomDataAndTypes 
            output realTypes
            [[5L; 5L]; [5L; 5L]; [5L; 5L]; [5L; 5L]] 
            (fun [a; b; c; d] -> UExpr.ifThenElse ((a <<== b) &&&& (b >>>> c)) (d) (a)) 

    [<TraceCompareFact>]    
    let ``Gather 1`` () =
        requireEqualTraces output (fun ctx ->
            let a = Var<single> (ctx / "a", [Size.fix 4L; Size.fix 3L])
            let i0 = Var<int64> (ctx / "i0", [Size.broadcastable; Size.fix 3L])
            let i1 = Var<int64> (ctx / "i1", [Size.broadcastable; Size.fix 3L])
            let expr = Expr a |> Expr.gather [Some (Expr i0); Some (Expr i1)]

            let av = HostTensor.counting 12L |> Tensor.reshape [4L; 3L] |> Tensor.convert
            let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
            let i1v = [0L; 0L; 1L] |> HostTensor.ofList |> Tensor.padLeft
            let varEnv = 
                VarEnv.empty
                |> VarEnv.add a (Tensor.transfer ctx.Dev av)
                |> VarEnv.add i0 (Tensor.transfer ctx.Dev i0v)
                |> VarEnv.add i1 (Tensor.transfer ctx.Dev i1v)
            expr.Untyped, varEnv)

    [<TraceCompareFact>]    
    let ``Gather 2`` () =
        requireEqualTraces output (fun ctx ->
            let a = Var<single> (ctx / "a", [Size.fix 4L; Size.fix 3L])
            let i0 = Var<int64> (ctx / "i0", [Size.broadcastable; Size.fix 3L])
            let expr = Expr a |> Expr.gather [Some (Expr i0); None]

            let av = HostTensor.counting 12L |> Tensor.reshape [4L; 3L] |> Tensor.convert
            let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
            let varEnv = 
                VarEnv.empty
                |> VarEnv.add a (Tensor.transfer ctx.Dev av)
                |> VarEnv.add i0 (Tensor.transfer ctx.Dev i0v)
            expr.Untyped, varEnv)

    [<TraceCompareFact>]    
    let ``Scatter 1`` () =
        requireEqualTraces output (fun ctx ->
            let a = Var<single> (ctx / "a", [Size.fix 4L; Size.fix 3L])
            let i0 = Var<int64> (ctx / "i0", [Size.broadcastable; Size.fix 3L])
            let shp = [Size.fix 5L; Size.fix 5L]
            let expr = Expr a |> Expr.scatter [Some (Expr i0); None] shp

            let av = HostTensor.counting 12L |> Tensor.reshape [4L; 3L] |> Tensor.convert
            let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
            let varEnv = 
                VarEnv.empty
                |> VarEnv.add a (Tensor.transfer ctx.Dev av)
                |> VarEnv.add i0 (Tensor.transfer ctx.Dev i0v)
            expr.Untyped, varEnv)

    [<TraceCompareFact>]    
    let ``Scatter 2`` () =
        requireEqualTraces output (fun ctx ->
            let a = Var<single> (ctx / "a", [Size.fix 4L; Size.fix 3L])
            let i0 = Var<int64> (ctx / "i0", [Size.broadcastable; Size.fix 3L])
            let i1 = Var<int64> (ctx / "i1", [Size.broadcastable; Size.fix 3L])
            let shp = [Size.fix 5L; Size.fix 5L]
            let expr = Expr a |> Expr.scatter [Some (Expr i0); Some (Expr i1)] shp

            let av = HostTensor.counting 12L |> Tensor.reshape [4L; 3L] |> Tensor.convert
            let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
            let i1v = [0L; 0L; 0L] |> HostTensor.ofList |> Tensor.padLeft
            let varEnv = 
                VarEnv.empty
                |> VarEnv.add a (Tensor.transfer ctx.Dev av)
                |> VarEnv.add i0 (Tensor.transfer ctx.Dev i0v)
                |> VarEnv.add i1 (Tensor.transfer ctx.Dev i1v)
            expr.Untyped, varEnv)

    [<TraceCompareFact>]    
    let ``Counting`` () =
        requireEqualTraces output (fun ctx ->
            let expr = Expr<_>.counting ctx.Dev (Size.fix 100L)
            expr.Untyped, VarEnv.empty)
