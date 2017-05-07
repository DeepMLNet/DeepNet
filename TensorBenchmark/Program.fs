open Tensor
open System.Diagnostics


[<EntryPoint>]
let main argv = 
    let shape = [10000L; 1000L]
    printfn "Shape: %A" shape

    let a : Tensor<single> = HostTensor.zeros shape
    let b : Tensor<single> = HostTensor.ones shape
    let mutable c : Tensor<single> = HostTensor.zeros shape
    let mutable d : Tensor<bool> = HostTensor.falses shape
    
    a.[[0L; 0L]] <- -1.0f
    a.[[0L; 1L]] <- 3.0f
    b.[[0L; 0L]] <- 2.0f

    let iters = 30

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a + b
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Plus time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- abs a
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Abs time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- sin a
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Sin time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- exp a
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Exp time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- sqrt a
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Sqrt time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- sgn a
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Sgn time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a / b
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Divide time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a % b
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Modulo time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a ** b
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Power time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        d <- a <<<< b
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Less time per iteration: %.3f ms" timePerIter

    printfn "a:\n%A" a
    printfn "b:\n%A" b
    printfn "c:\n%A" c
    printfn "d:\n%A" d

    printfn "sgn -1.0f: %f   sgn 0.0f: %f" (sgn -1.0f) (sgn 0.0f)

    printfn "ndims a: %d" (Tensor.nDims a)

    let m : Tensor<bool> = HostTensor.falses shape
    let n : Tensor<bool> = HostTensor.trues shape
    let mutable k : Tensor<bool> = HostTensor.falses shape
    
    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        k <- ~~~~m
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Not time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        k <- m &&&& n
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "And time per iteration: %.3f ms" timePerIter

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        k <- m |||| n
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Or time per iteration: %.3f ms" timePerIter

    printfn "m:\n%A" m
    printfn "n:\n%A" n
    printfn "k:\n%A" k

    0 
