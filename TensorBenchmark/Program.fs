open Tensor
open System.Diagnostics



let testCuda () =
    let shape = [5L; 5L]

    //Tensor.Cuda.Backend.Cfg.DebugCompile <- true
    Tensor.Cuda.Backend.Cfg.GatherScatterStacktrace <- true

    let a = HostTensor.ones<single> shape
    printfn "a=\n%A" a

    printfn "copy to cuda..."
    let ca = a |> CudaTensor.transfer
    printfn "ca=\n%A" ca.Full
    printfn "copy to host..."
    let ha = ca |> HostTensor.transfer
    printfn "ha=\n%A" ha

    printfn "copy cuda to cuda..."
    let cb = Tensor.copy ca
    printfn "cb=\n%A" cb

    printfn "fill cuda..."
    cb.FillConst(-5.5f)
    printfn "cb=\n%A" cb

    printfn "cuda abs..."
    let cb = abs cb
    printfn "cb=\n%A" cb

    printfn "cuda ca + cb..."
    let cc = ca + cb
    printfn "cc=\n%A" cc

    printfn "cuda sumAxis cc..."
    let ci = Tensor.sumAxis 0 cc
    printfn "ci=\n%A" ci

    printfn "cuda maxAxis cc..."
    let cg = Tensor.maxAxis 0 cc
    printfn "cg=\n%A" cg

    printfn "cuda argMinAxis cc..."
    let cj = Tensor.argMinAxis 1 cc
    printfn "cj=\n%A" cj

    printfn "cuda convert to int..."
    let cgInt = Tensor.convert<int> cg
    printfn "cgInt=\n%A" cgInt

    printfn "cuda ca <<== cb..."
    let cd = ca <<== cb
    printfn "cd=\n%A" cd

    printfn "not cd..."
    let ce = ~~~~cd
    printfn "ce=\n%A" ce

    printfn "ifthenelse..."
    let ck = Tensor.ifThenElse ce ca cc
    printfn "ck=\n%A" ck

    printfn "cd or ce..."
    let cf = cd |||| ce
    printfn "cf=\n%A" cf

    printfn "all cf..."
    let ch = Tensor.allAxis 0 cf
    printfn "ch=\n%A" ch

    printfn "cuda gather..."
    let idxs0 = CudaTensor.zeros<int64> [3L; 3L]
    let cgather = Tensor.gather [Some idxs0; None] cb
    printfn "cgather=\n%A" cgather

    printfn "cuda scatter..."
    let idxs1 = CudaTensor.ones<int64> shape
    let cscatter = Tensor.scatter [Some idxs1; None] [6L; 6L] cb
    printfn "cscatter=\n%A" cscatter


[<EntryPoint>]
let main argv = 
    Tensor.Utils.Util.disableCrashDialog ()
    testCuda()
    exit 0


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
        c <- a.T .* b
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Dot time per iteration: %.3f ms" timePerIter

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
