namespace global
#nowarn "25"

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Expr.Ops
open Tensor.Cuda
open TestUtils



type OpTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let realTypes = [typeof<single>; typeof<double>]

    [<Fact>]    
    let ``Singular matrix inverse exception`` () =
        runOnAllDevs output (fun ctx ->
            match ctx.Dev with
            | :? TensorCudaDevice -> Cuda.Cfg.Stacktrace <- true
            | _ -> ()

            let a = Var<single> (ctx / "a", [Size.fix 3L; Size.fix 3L])
            let expr = Expr.invert (Expr a)
            let av = HostTensor.zeros<single> [3L; 3L]
            let varEnv =
                VarEnv.empty
                |> VarEnv.add a (Tensor.transfer ctx.Dev av)
            printfn "a=\n%A" av
            try
                printfn "a^-1=\n%A" (expr |> Expr.eval varEnv)
                failwith "No exception encountered"
            with :? Tensor.SingularMatrixException ->
                printfn "Singular matrix exception"

            match ctx.Dev with
            | :? TensorCudaDevice -> Cuda.Cfg.Stacktrace <- false
            | _ -> ())

    [<Fact>]
    let ``Comparison, logics, conditionals`` () =
        runOnAllDevs output (fun ctx ->
            let a = Var<single> (ctx / "a", [Size.fix 3L; Size.fix 3L])
            let b = Var<single> (ctx / "b", [Size.fix 3L; Size.fix 3L])
            let c = Var<single> (ctx / "c", [Size.fix 3L; Size.fix 3L])
            let d = Var<single> (ctx / "d", [Size.fix 3L; Size.fix 3L])
            let expr = Expr.ifThenElse ((Expr a <<== Expr b) &&&& (Expr b >>>> Expr c)) (Expr d) (Expr a) 

            let rng = System.Random (123)
            let av = HostTensor.randomUniform rng (-1.0f, 1.0f) [3L; 3L] |> Tensor.transfer ctx.Dev
            let bv = HostTensor.randomUniform rng (-1.0f, 1.0f) [3L; 3L] |> Tensor.transfer ctx.Dev
            let cv = HostTensor.randomUniform rng (-1.0f, 1.0f) [3L; 3L] |> Tensor.transfer ctx.Dev
            let dv = HostTensor.randomUniform rng (-1.0f, 1.0f) [3L; 3L] |> Tensor.transfer ctx.Dev
            let varEnv = VarEnv.ofSeq [a, av; b, bv; c, cv; d, dv]

            let res = expr |> Expr.eval varEnv
            printfn "res=%A" res)



    //let checkFiniteOpTest diagVal offDiagVal =
    //    let a = Expr.var<single> "a" [SizeSpec.fix 3L; SizeSpec.fix 3L]
    //    let b = Expr.var<single> "b" [SizeSpec.fix 3L; SizeSpec.fix 3L]
    //    let expr = a / b |> Expr.checkFinite "a / b"
    //    let fn = Func.make<single> DevCuda.DefaultFactory expr |> arg2 a b
    //    let av = CudaTensor.ones<single> [3L; 3L]
    //    let dv = diagVal * HostTensor.ones<single> [3L] |> CudaTensor.transfer
    //    let bv = offDiagVal * HostTensor.ones<single> [3L; 3L] |> CudaTensor.transfer
    //    (Tensor.diag bv).[*] <- dv
    //    printfn "a=\n%A" av
    //    printfn "b=\n%A" bv
    //    let iav = fn av bv
    //    printfn "a / b=\n%A" iav

    //[<Fact>]
    
    //let ``Check finite on CUDA failing`` () =
    //    SymTensor.Compiler.Cuda.Debug.TerminateWhenNonFinite <- false
    //    printfn "failing:"
    //    checkFiniteOpTest 1.0f 0.0f

    //[<Fact>]
    
    //let ``Check finite on CUDA passing`` () =
    //    printfn "passing:"
    //    checkFiniteOpTest 1.0f 0.5f

    //[<Fact>]
    //let ``ReverseAxis on host`` () =
    //    let a = Expr.var<int> "a" [SizeSpec.fix 3L; SizeSpec.fix 2L]
    //    let expr0 = Expr.reverseAxis 0 a
    //    let expr1 = Expr.reverseAxis 1 a
    //    let fn = Func.make2<int, int> DevHost.DefaultFactory expr0 expr1 |> arg1 a

    //    let av = [0 .. 5] |> HostTensor.ofList |> Tensor.reshape [3L; 2L]
    //    printfn "av=\n%A" av

    //    let rav0, rav1 = fn av
    //    printfn "rev 0 av=\n%A" rav0
    //    printfn "rev 1 av=\n%A" rav1




    //[<Fact>]
    //let ``Replicate`` () =
    //    let a = Expr.var<single> "a" [SizeSpec.fix 2L; SizeSpec.fix 3L]
    //    let expr0 = Expr.replicate 0 (SizeSpec.fix 2L) a
    //    let expr1 = Expr.replicate 1 (SizeSpec.fix 3L) a
    //    let fns = Func.make2<single, single> DevCuda.DefaultFactory expr0 expr1 |> arg1 a
    //    let av = [[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]] |> HostTensor.ofList2D 
    //    let av0, av1 = fns av
    //    printfn "a=\n%A" av 
    //    printfn "rep 0 2 a=\n%A" av0
    //    printfn "rep 1 3 a=\n%A" av1

    //[<Fact>]    
    //let ``ReplicateTo on CUDA`` () =
    //    let a = Expr.var<single> "a" [SizeSpec.fix 2L; SizeSpec.fix 3L]
    //    let expr0 = Expr.replicateTo 0 (SizeSpec.fix 6L) a
    //    let expr1 = Expr.replicateTo 1 (SizeSpec.fix 7L) a
    //    let fns = Func.make2<single, single> DevCuda.DefaultFactory expr0 expr1 |> arg1 a
    //    let av = [[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]] |> HostTensor.ofList2D 
    //    let av0, av1 = fns av
    //    printfn "a=\n%A" av 
    //    printfn "repTo 0 6 a=\n%A" av0
    //    printfn "repTo 1 7 a=\n%A" av1

    //[<Fact>]    
    //let ``Derivative of ReplicateTo on CUDA`` () =
    //    let a = Expr.var<single> "a" [SizeSpec.fix 2L; SizeSpec.fix 3L]
    //    let expr0 = Expr.replicateTo 0 (SizeSpec.fix 6L) a
    //    let expr1 = Expr.replicateTo 1 (SizeSpec.fix 7L) a
    //    let da0 = Deriv.compute expr0 |> Deriv.ofVar a
    //    let da1 = Deriv.compute expr1 |> Deriv.ofVar a
    //    let fns = Func.make2<single, single> DevCuda.DefaultFactory da0 da1 |> arg1 a
    //    let av = [[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]] |> HostTensor.ofList2D 
    //    let dav0, dav1 = fns av
    //    printfn "a=\n%A" av 
    //    printfn "d(repTo 0 7 a) / da=\n%A" dav0.Full
    //    printfn "d(repTo 1 5 a) / da=\n%A" dav1.Full

    //[<Fact>]
    //let ``Derivative of ReplicateTo on host`` () =
    //    let a = Expr.var<single> "a" [SizeSpec.fix 2L; SizeSpec.fix 3L]
    //    let expr0 = Expr.replicateTo 0 (SizeSpec.fix 6L) a
    //    let expr1 = Expr.replicateTo 1 (SizeSpec.fix 7L) a
    //    let da0 = Deriv.compute expr0 |> Deriv.ofVar a
    //    let da1 = Deriv.compute expr1 |> Deriv.ofVar a
    //    let fns = Func.make2<single, single> DevHost.DefaultFactory da0 da1 |> arg1 a
    //    let av = [[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]] |> HostTensor.ofList2D 
    //    let dav0, dav1 = fns av
    //    printfn "a=\n%A" av 
    //    printfn "d(repTo 0 7 a) / da=\n%A" dav0.Full
    //    printfn "d(repTo 1 5 a) / da=\n%A" dav1.Full