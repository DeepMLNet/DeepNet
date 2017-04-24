open ArrayNDNS
open System.Diagnostics


[<EntryPoint>]
let main argv = 
    let shape = [10000L; 10000L]
    let a : Tensor<single> = HostTensor.zeros shape
    let b : Tensor<single> = HostTensor.zeros shape
    let mutable c : Tensor<single> = HostTensor.zeros shape
    
    a.[[0L; 0L]] <- 1.0f
    b.[[0L; 0L]] <- 2.0f

    let iters = 10

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a + b
        //c.[[0L; 0L]] <- 0.0f
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Time per iteration: %.3f ms" timePerIter


    printfn "a: %A" (a.Storage :?> TensorHostStorage<single>).Data
    printfn "b: %A" (b.Storage :?> TensorHostStorage<single>).Data
    printfn "c: %A" (c.Storage :?> TensorHostStorage<single>).Data

    0 
