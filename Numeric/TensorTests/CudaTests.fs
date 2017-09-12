module CudaTests

open Xunit
open FsUnit.Xunit

open Tensor.Utils
open Tensor


/// Test that only runs when CUDA is available.
type CudaFactAttribute() as this =
    inherit FactAttribute()
    do
        try ignore Cuda.context
        with err ->
            this.Skip <- "CUDA not present"


[<CudaFact>]
let ``Tensor transfer to Cuda``() =   
    let data = HostTensor.counting 30L |> Tensor.float |> Tensor.reshape [3L; 10L]
    let cuda = CudaTensor.transfer data
    let back = HostTensor.transfer cuda

    printfn "data:\n%A" data
    printfn "back:\n%A" back

    Tensor.almostEqual data back |> should equal true


[<CudaFact>]
let ``Tensor transfer to Cuda 2``() =    
    let data = HostTensor.counting 30L |> Tensor.float |> Tensor.reshape [3L; 2L; 5L]
    let data = data.Copy (order=CustomOrder [1; 0; 2])
    printfn "data layout:%A" data.Layout

    let cuda = CudaTensor.transfer data
    let back = HostTensor.transfer cuda

    printfn "data:\n%A" data
    printfn "back:\n%A" back

    Tensor.almostEqual data back |> should equal true


[<CudaFact>]
let ``Mixed Cuda tests`` () =
    // TODO: this needs cleanup

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
