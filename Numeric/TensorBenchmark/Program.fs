open System.Diagnostics
open System

open Tensor
open Tensor.Utils


let testCuda () =
    let shape = [5L; 5L]

    //Tensor.Cuda.Backend.Cfg.DebugCompile <- true
    Tensor.Cuda.Backend.Cfg.Stacktrace <- true

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

    printfn "cuda diagonal * 3..."
    let cdiag = 3.0f * CudaTensor.identity 3L
    printfn "cdiag=\n%A" cdiag

    printfn "cuda invert..."
    //let cinv = Tensor.invert cb
    let cinv = Tensor.invert cdiag
    printfn "cinv=\n%A" cinv

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

    printfn "cuda dot..."
    let c1 = cscatter .* cscatter
    printfn "c1=\n%A" c1

    printfn "cuda replicate..."
    let cscatterRep = Tensor.replicate 0 2L cscatter.[NewAxis, *, *]
    printfn "cscatterRep=\n%A" cscatterRep

    printfn "cuda batched dot..."
    let c2 = cscatterRep .* cscatterRep
    printfn "c2=\n%A" c2


[<EntryPoint>]
let main argv = 
    Tensor.Utils.Util.disableCrashDialog ()      
    let dev = 
        match List.ofArray argv with
        | ["cuda"] -> CudaTensor.Dev
        | ["host"] | [] -> HostTensor.Dev
        | _ -> 
            printfn "unknown device"
            exit 1
    let shape = [10000L; 1000L]

    let sync () =
        if dev = CudaTensor.Dev then
            Cuda.context.Synchronize()

    let collect () =
        ()

    printfn "On device %A" dev
    printfn "Shape: %A" shape

    let a : Tensor<single> = Tensor.zeros dev shape
    let b : Tensor<single> = Tensor.ones dev shape
    let mutable c : Tensor<single> = Tensor.zeros dev shape
    let mutable d : Tensor<bool> = Tensor.falses dev shape
    
    a.[[0L; 0L]] <- -1.0f
    a.[[0L; 1L]] <- 3.0f
    b.[[0L; 0L]] <- 2.0f

    let iters = 30

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a + b
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Plus time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- abs a
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Abs time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a.T .* b
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Dot time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- sin a
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Sin time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- exp a
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Exp time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- sqrt a
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Sqrt time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- sgn a
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Sgn time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a / b
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Divide time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a % b
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Modulo time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        c <- a ** b
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Power time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        d <- a <<<< b
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Less time per iteration: %.3f ms" timePerIter
    collect()

    //printfn "a:\n%A" a
    //printfn "b:\n%A" b
    //printfn "c:\n%A" c
    //printfn "d:\n%A" d

    //printfn "sgn -1.0f: %f   sgn 0.0f: %f" (sgn -1.0f) (sgn 0.0f)

    //printfn "ndims a: %d" (Tensor.nDims a)

    let m : Tensor<bool> = Tensor.falses dev shape
    let n : Tensor<bool> = Tensor.trues dev shape
    let mutable k : Tensor<bool> = Tensor.falses dev shape
    
    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        k <- ~~~~m
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Not time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        k <- m &&&& n
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "And time per iteration: %.3f ms" timePerIter
    collect()

    let startTime = Stopwatch.StartNew()
    for i in 1 .. iters do
        k <- m |||| n
    sync ()
    let duration = startTime.ElapsedMilliseconds
    let timePerIter = float duration / float iters
    printfn "Or time per iteration: %.3f ms" timePerIter
    collect()

    //printfn "m:\n%A" m
    //printfn "n:\n%A" n
    //printfn "k:\n%A" k

    0 
