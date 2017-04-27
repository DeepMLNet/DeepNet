open ArrayNDNS
open System.Diagnostics


[<EntryPoint>]
let main argv = 
    let shape = [10000L; 1000L]
    let a : Tensor<single> = HostTensor.zeros shape
    let b : Tensor<single> = HostTensor.ones shape
    let mutable c : Tensor<single> = HostTensor.zeros shape
    
    a.[[0L; 0L]] <- 1.0f
    b.[[0L; 0L]] <- 2.0f

    let _t= a.[0L,Fill]

    let iters = 30

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a + b
        //c.[[0L; 0L]] <- 0.0f
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Time per iteration: %.3f ms" timePerIter


    printfn "a:\n%A" a
    printfn "b:\n%A" b
    printfn "c:\n%A" c

    0 
