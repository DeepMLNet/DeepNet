namespace global

#nowarn "25"

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit
open System.Threading

open Tensor.Utils
open Tensor


/// Test that only runs when CUDA is available.
type CudaFactAttribute() as this =
    inherit FactAttribute()
    do
        if CudaDev.count = 0 then
            this.Skip <- "CudaDev.count = 0"


type CudaTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<CudaFact>]
    let ``Device info`` () =
        for i, info in Seq.indexed CudaDev.info do
            printfn "Cuda device %d: %A" i info.DeviceName

    [<CudaFact>]
    let ``Tensor device for each GPU`` () =
        for i in 0 .. CudaDev.count-1 do
            printfn "Getting tensor device for GPU %d" i
            let dev = CudaDev.get i
            printfn "Done: %A" dev

    [<CudaFact>]
    let ``Create`` () =
        let a = CudaTensor.ones<single> [5L; 5L]
        printfn "a=\n%A" a

    [<CudaFact>]
    let ``Tensor transfer to Cuda``() =   
        let data = HostTensor.counting 30L |> Tensor<float>.convert |> Tensor.reshape [3L; 10L]
        let cuda = CudaTensor.transfer data
        let back = HostTensor.transfer cuda

        printfn "data:\n%A" data
        printfn "back:\n%A" back

        Tensor.almostEqual (data, back) |> should equal true

    [<CudaFact>]
    let ``Tensor transfer to Cuda 2``() =    
        let data = HostTensor.counting 30L |> Tensor<float>.convert |> Tensor.reshape [3L; 2L; 5L]
        let data = Tensor.copy (data, order=CustomOrder [1; 0; 2])
        printfn "data layout:%A" data.Layout

        let cuda = CudaTensor.transfer data
        let back = HostTensor.transfer cuda

        printfn "data:\n%A" data
        printfn "back:\n%A" back

        Tensor.almostEqual (data, back)  |> should equal true

    [<CudaFact>]
    let ``Multi-threaded Cuda``() =
        let repetitions = 5
        let nThreads = 5

        let threadFn i () = 
            Thread.Sleep(100)
            for r in 1..repetitions do
                printfn "Thread %d repetition %d" i r
                let a: Tensor<float32> = CudaTensor.zeros [100L]
                let b = CudaTensor.ones [100L]
                let c = a + b
                Tensor.almostEqual (b, c) |> should equal true
            printfn "Thread %d done" i
         
        printfn "Starting threads..."
        let threads = [
            for i in 1..nThreads do
                let thread = Thread (threadFn i)
                thread.Start()
                yield thread
        ]

        printfn "Waiting for threads..."
        for t in threads do
            t.Join (3000) |> should equal true

    [<CudaFact>]
    let ``Single matrix dot`` () =
        let h = HostTensor.init [5L; 3L] (fun [|i; j|] -> 3.0f * single i + single j)
        let i = 0.1f + HostTensor.identity 3L
        let hi = h .* i

        let hGpu = CudaTensor.transfer h
        let iGpu = CudaTensor.transfer i
        let hiGpu = hGpu .* iGpu

        Tensor.almostEqual (hi, HostTensor.transfer hiGpu) |> should equal true

    [<CudaFact>]
    let ``Double matrix dot`` () =
        let h = HostTensor.init [5L; 3L] (fun [|i; j|] -> 3.0 * double i + double j)
        let i = 0.1 + HostTensor.identity 3L
        let hi = h .* i

        let hGpu = CudaTensor.transfer h
        let iGpu = CudaTensor.transfer i
        let hiGpu = hGpu .* iGpu

        Tensor.almostEqual (hi, HostTensor.transfer hiGpu) |> should equal true

    [<CudaFact>]
    let ``Mixed Cuda tests`` () =
        // TODO: this needs cleanup

        let shape = [5L; 5L]

        //Tensor.Cuda.Backend.Cfg.DebugCompile <- true
        Tensor.Cuda.Cfg.Stacktrace <- true

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
        let cgInt = Tensor<int>.convert cg
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
