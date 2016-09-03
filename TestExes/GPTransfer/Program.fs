open ArrayNDNS
open SymTensor



//let testSquaredExponentialCovarianceMatrix () =
//    let nGps = SizeSpec.symbol "nGps"
//    let nTrnSmpls = SizeSpec.symbol "nTrnSmpls"
//
//    let trnX = Expr.var<single> "trnX" [nGps; nTrnSmpls]
//    let lengthscale = Expr.var "lengthscale" [nGps]
//
//    let cm = GPTransferOps.squaredExponentialCovarianceMatrix trnX lengthscale
//    let cmFn = Func.make DevHost.DefaultFactory cm |> arg2 trnX lengthscale
//
//    let derivs = Deriv.compute cm
//    let cmWrtTrnX = derivs |> Deriv.ofVar trnX
//    let cmWrtLengthscale = derivs |> Deriv.ofVar lengthscale
//    let derivFn = Func.make2 DevHost.DefaultFactory cmWrtTrnX cmWrtLengthscale |> arg2 trnX lengthscale
//
//    let trnXVal = [[1.0f; 1.1f; 2.0f]] |> ArrayNDHost.ofList2D
//    let lengthscaleVal = [0.5f] |> ArrayNDHost.ofList
//    let cmVal = cmFn trnXVal lengthscaleVal
//    let cmWrtTrnXVal, cmWrtLengthscaleVal = derivFn trnXVal lengthscaleVal 
//
//    printfn "trnXVal=\n%A" trnXVal
//    printfn "lengthscaleVal=\n%A" lengthscaleVal
//    printfn "cmVal=\n%A" cmVal
//    printfn ""
//    printfn "d(cmVal)/d(trnXVal)=\n%A" cmWrtTrnXVal
//    printfn "d(cmVal)/d(lengthscaleVal)=\n%A" cmWrtLengthscaleVal
    

[<EntryPoint>]
let main argv = 
    //MathInterface.doMathTest ()
    //MathInterface.doMathTest2 ()

    //testSquaredExponentialCovarianceMatrix ()

    0


