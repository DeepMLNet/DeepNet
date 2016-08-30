open ArrayNDNS
open SymTensor



let testSquaredExponentialCovarianceMatrix () =
    let nGps = SizeSpec.symbol "nGps"
    let nTrnSmpls = SizeSpec.symbol "nTrnSmpls"

    let trnX = Expr.var<single> "trnX" [nGps; nTrnSmpls]
    let lengthscale = Expr.var "lengthscale" [nGps]

    let cm = GPTransferOps.squaredExponentialCovarianceMatrix trnX lengthscale
    let cmFn = Func.make DevHost.DefaultFactory cm |> arg2 trnX lengthscale


    let trnXVal = [[1.0f; 1.1f; 2.0f]] |> ArrayNDHost.ofList2D
    let lengthscaleVal = [0.5f] |> ArrayNDHost.ofList
    let cmVal = cmFn trnXVal lengthscaleVal

    printfn "trnXVal=\n%A" trnXVal
    printfn "lengthscaleVal=\n%A" lengthscaleVal
    printfn "cmVal=\n%A" cmVal



[<EntryPoint>]
let main argv = 
    //MathInterface.doMathTest ()
    //MathInterface.doMathTest2 ()

    testSquaredExponentialCovarianceMatrix ()

    0


