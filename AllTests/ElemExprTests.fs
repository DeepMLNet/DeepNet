module ElemExprTests
#nowarn "25"

open Xunit
open FsUnit.Xunit

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open TestUtils


[<Fact>]
let ``Eval: simple`` () =   
    // input  x[i, j]
    //        y[m, n]
    // output  [k]

    printfn "======= Testing evaluation:"

    let k = ElemExpr.idx 0   
    let x = ElemExpr.argElem 0
    let y = ElemExpr.argElem 1
    let expr = 2.0 * (x [k; k]) + y [SizeSpec.zero; k]

    let xVal = [1.0; 2.0; 3.0] |> ArrayNDHost.ofList |> ArrayND.diagMat
    let yVal = [[4.0; 5.0; 6.0]
                [7.0; 8.0; 9.0]] |> ArrayNDHost.ofList2D
    let res = ElemExpr.eval expr [xVal; yVal] [xVal.Shape.[0]]

    printfn "Expr:\n%A" expr
    printfn "x=\n%A" xVal
    printfn "y=\n%A" yVal
    printfn "result=\n%A" res

    let expected = [6.0; 9.0; 12.0] |> ArrayNDHost.ofList
    ArrayND.almostEqual res expected |> ArrayND.value |> should equal true


[<Fact>]
let ``Eval: sum`` () =   
    // input  x[i, j]
    // output  [k]

    printfn "======= Testing evaluation sum:"

    let is = SizeSpec.fix 3
    let js = SizeSpec.fix 3

    let k = ElemExpr.idx 0   
    let x = ElemExpr.argElem 0
    let l = ElemExpr.sumIdx "l"
    let expr = 2.0 * (ElemExpr.sum l SizeSpec.zero (is-1) (x [l; k]))

    let xVal = [1.0; 2.0; 3.0] |> ArrayNDHost.ofList |> ArrayND.diagMat
    let xVal = xVal + ArrayNDHost.ones [3; 3]
    let res = ElemExpr.eval expr [xVal] [xVal.Shape.[0]]

    printfn "Expr:\n%A" expr
    printfn "x=\n%A" xVal
    printfn "result=\n%A" res

    let expected = [8.0; 10.0; 12.0] |> ArrayNDHost.ofList
    ArrayND.almostEqual res expected |> ArrayND.value |> should equal true

[<Fact>]
let ``Deriv: 1`` () =   
    // input  x[i, j]
    //        y[m, n]
    // output  [k]

    printfn "======= Testing derivatives:"

    let ks = SizeSpec.symbol "ks"

    let k = ElemExpr.idx 0   
    let x = ElemExpr.argElem 0
    let y = ElemExpr.argElem 1
    let expr = 2.0 * (x [k; k]) + y [SizeSpec.zero; k]
    let dExpr = ElemExprDeriv.buildDerivElemExpr expr [ks] 2

    printfn "Expr:\n%A" expr
    printfn "dExpr / dx:\n%A" dExpr.[0]
    printfn "dExpr / dy:\n%A" dExpr.[1]


[<Fact>]
let ``Deriv: 2`` () =   
    // input  x[i, j]
    //        y[m, n]
    // output  [k, l]

    printfn "======= Testing derivatives 2:"

    let ks = SizeSpec.symbol "ks"
    let ls = SizeSpec.symbol "ls"
    let k = ElemExpr.idx 0   
    let l = ElemExpr.idx 1

    let x = ElemExpr.argElem 0
    let y = ElemExpr.argElem 1
    let expr = 2.0 * (x [k; k]) + y [l; k]
    let dExpr = ElemExprDeriv.buildDerivElemExpr expr [ks; ls] 2

    printfn "Expr:\n%A" expr
    printfn "dExpr / dx:\n%A" dExpr.[0]
    printfn "dExpr / dy:\n%A" dExpr.[1]


[<Fact>]
let ``Deriv: sum`` () =   
    // input  x[i, j]
    // output  [k]

    printfn "======= Testing derivative sum:"

    let is = SizeSpec.fix 3
    let js = SizeSpec.fix 3

    let k = ElemExpr.idx 0   
    let x = ElemExpr.argElem 0
    let l = ElemExpr.sumIdx "l"
    let expr = 2.0 * (ElemExpr.sum l SizeSpec.zero (is-1) (x [l; k]))
    let dExpr = ElemExprDeriv.buildDerivElemExpr expr [is] 1

    printfn "Expr:\n%A" expr
    printfn "dExpr/dx:\n%A" dExpr.[0]
   

[<Fact>]
let ``Eval and deriv: KSE`` () =   
    // input  x[gp, smpl]
    //        l[gp]
    // output cov[gp, smpl1, smpl2]

    printfn "======= Testing KSE:"

    let nGps = SizeSpec.fix 1
    let nSmpls = SizeSpec.fix 3
    let gp = ElemExpr.idx 0   
    let smpl1 = ElemExpr.idx 1
    let smpl2 = ElemExpr.idx 2

    let x = ElemExpr.argElem 0
    let l = ElemExpr.argElem 1
    let kse = exp (- ((x [gp; smpl1] - x [gp; smpl2])**2.0) / (2.0 * (l [gp])**2.0) )
    let dKse = ElemExprDeriv.buildDerivElemExpr kse [nGps; nSmpls; nSmpls] 2

    printfn "KSE:\n%A" kse
    printfn "dKSE / dx:\n%A" dKse.[0]
    printfn "dKSE / dl:\n%A" dKse.[1]


    let xVal = [[1.0; 1.1; 2.0]] |> ArrayNDHost.ofList2D
    let lVal = [0.5] |> ArrayNDHost.ofList
    let kseVal = ElemExpr.eval kse [xVal; lVal] [1; 3; 3]

    printfn "x=\n%A" xVal
    printfn "l=\n%A" lVal
    printfn "kse=\n%A" kseVal

    let dKseVal = kseVal |> ArrayND.reshape [1; 1; 3; 3]

    let dKSe0Val = ElemExpr.eval dKse.[0] [xVal; lVal; dKseVal] [1; 1; 3]
    let dKSe1Val = ElemExpr.eval dKse.[1] [xVal; lVal; dKseVal] [1; 1]
    printfn "dkse / dx=\n%A" dKSe0Val
    printfn "dkse / dl=\n%A" dKSe1Val


[<Fact>]
let ``Codegen: KSE`` () =  
    printfn "======= Testing KSE codegen:"

    let nGps = SizeSpec.fix 1
    let nSmpls = SizeSpec.fix 3
    let gp = ElemExpr.idx 0   
    let smpl1 = ElemExpr.idx 1
    let smpl2 = ElemExpr.idx 2

    let x = ElemExpr.argElem 0
    let l = ElemExpr.argElem 1
    let kse = exp (- ((x [gp; smpl1] - x [gp; smpl2])**2.0) / (2.0 * (l [gp])**2.0) )
    let dKse = ElemExprDeriv.buildDerivElemExpr kse [nGps; nSmpls; nSmpls] 2
    let dKsedX, dKsedL = dKse.[0], dKse.[1]

    let uKse = UElemExpr.toUElemFunc kse 3 2
    let udKsedX = UElemExpr.toUElemFunc dKsedX (2+1) (2+1)
    let udKsedL = UElemExpr.toUElemFunc dKsedL (1+1) (2+1)
    let kseCode = CudaElemExpr.generateFunctor "KSE" uKse
    let dKsedXCode =  CudaElemExpr.generateFunctor "dKSEdX" udKsedX 
    let dKsedLCode =  CudaElemExpr.generateFunctor "dKSEdL" udKsedL 

    printfn "Code:\n%s%s%s" kseCode dKsedXCode dKsedLCode


[<Fact>]
let ``Eval and deriv: KSE in Expr on Host`` () =   
    // input  x[gp, smpl]
    //        l[gp]
    // output cov[gp, smpl1, smpl2]

    printfn "======= Testing KSE in Expr on Host:"

    let nGps = SizeSpec.symbol "nGps"
    let nSmpls = SizeSpec.symbol "nSmpls"
    let gp = ElemExpr.idx 0   
    let smpl1 = ElemExpr.idx 1
    let smpl2 = ElemExpr.idx 2

    let x = ElemExpr.argElem 0
    let l = ElemExpr.argElem 1
    let kseExpr = exp (- ((x [gp; smpl1] - x [gp; smpl2])**2.0) / (2.0 * (l [gp])**2.0) )

    let xTensor = Expr.var "xTensor" [nGps; nSmpls] 
    let lTensor = Expr.var "lTensor" [nGps]
    let kse = Expr.elements [nGps; nSmpls; nSmpls] kseExpr [xTensor; lTensor]

    let dKse = Deriv.compute kse
    let dKsedX = dKse |> Deriv.ofVar xTensor
    let dKsedL = dKse |> Deriv.ofVar lTensor

    let kseFn = Func.make DevHost.DefaultFactory kse |> arg2 xTensor lTensor
    let dKseFn = Func.make2 DevHost.DefaultFactory dKsedX dKsedL |> arg2 xTensor lTensor

    let xVal = [[1.0; 1.1; 2.0]] |> ArrayNDHost.ofList2D
    let lVal = [0.5] |> ArrayNDHost.ofList

    let kseVal = kseFn xVal lVal
    let dKsedXVal, dKsedLVal = dKseFn xVal lVal

    printfn "x=\n%A" xVal
    printfn "l=\n%A" lVal
    printfn "kse=\n%A" kseVal
    printfn "dkse / dx=\n%A" dKsedXVal
    printfn "dkse / dl=\n%A" dKsedLVal

[<Fact>]
let ``Eval and deriv: KSE in Expr on CUDA`` () =   
    // input  x[gp, smpl]
    //        l[gp]
    // output cov[gp, smpl1, smpl2]

    printfn "======= Testing KSE in Expr on CUDA:"

    let nGps = SizeSpec.symbol "nGps"
    let nSmpls = SizeSpec.symbol "nSmpls"
    let gp = ElemExpr.idx 0   
    let smpl1 = ElemExpr.idx 1
    let smpl2 = ElemExpr.idx 2

    let x = ElemExpr.argElem 0
    let l = ElemExpr.argElem 1
    let kseExpr = exp (- ((x [gp; smpl1] - x [gp; smpl2])**2.0f) / (2.0f * (l [gp])**2.0f) )

    let xTensor = Expr.var "xTensor" [nGps; nSmpls] 
    let lTensor = Expr.var "lTensor" [nGps]
    let kse = Expr.elements [nGps; nSmpls; nSmpls] kseExpr [xTensor; lTensor]

    let dKse = Deriv.compute kse
    let dKsedX = dKse |> Deriv.ofVar xTensor
    let dKsedL = dKse |> Deriv.ofVar lTensor

    let kseFn = Func.make DevCuda.DefaultFactory kse |> arg2 xTensor lTensor
    let dKseFn = Func.make2 DevCuda.DefaultFactory dKsedX dKsedL |> arg2 xTensor lTensor

    let xVal = [[1.0f; 1.1f; 2.0f]] |> ArrayNDHost.ofList2D |> ArrayNDCuda.toDev
    let lVal = [0.5f] |> ArrayNDHost.ofList |> ArrayNDCuda.toDev

    let kseVal = kseFn xVal lVal
    let dKsedXVal, dKsedLVal = dKseFn xVal lVal

    printfn "x=\n%A" xVal
    printfn "l=\n%A" lVal
    printfn "kse=\n%A" kseVal
    printfn "dkse / dx=\n%A" dKsedXVal
    printfn "dkse / dl=\n%A" dKsedLVal


let kseElemExpr () = 
    // input  x[gp, smpl]
    //        l[gp]
    // output cov[gp, smpl1, smpl2]
    let gp = ElemExpr.idx 0   
    let smpl1 = ElemExpr.idx 1
    let smpl2 = ElemExpr.idx 2
    let x = ElemExpr.argElem 0
    let l = ElemExpr.argElem 1
    exp (- ((x [gp; smpl1] - x [gp; smpl2])**2.0f) / (2.0f * (l [gp])**2.0f) )

[<Fact>]
let ``Trace compare: KSE`` () =    
    let nGps = 2
    let nSmpls = 3
    requireEqualTracesWithRandomData [[nGps; nSmpls]; [nGps]] (fun [xTensor; lTensor] ->
        let elemExpr = kseElemExpr ()
        let kse = 
            Expr.elements [SizeSpec.fix nGps; SizeSpec.fix nSmpls; SizeSpec.fix nSmpls] 
                elemExpr [xTensor; lTensor]
        kse    
    )


[<Fact>]
let ``Trace compare: dKSE/dX`` () =    
    let nGps = 2
    let nSmpls = 3
    requireEqualTracesWithRandomData [[nGps; nSmpls]; [nGps]] (fun [xTensor; lTensor] ->
        let elemExpr = kseElemExpr ()
        let kse = 
            Expr.elements [SizeSpec.fix nGps; SizeSpec.fix nSmpls; SizeSpec.fix nSmpls] 
                elemExpr [xTensor; lTensor]
        let dKse = Deriv.compute kse
        dKse |> Deriv.ofVar xTensor
    )

[<Fact>]
let ``Trace compare: dKSE/dL`` () =    
    let nGps = 2
    let nSmpls = 3
    requireEqualTracesWithRandomData [[nGps; nSmpls]; [nGps]] (fun [xTensor; lTensor] ->
        let elemExpr = kseElemExpr ()
        let kse = 
            Expr.elements [SizeSpec.fix nGps; SizeSpec.fix nSmpls; SizeSpec.fix nSmpls] 
                elemExpr [xTensor; lTensor]
        let dKse = Deriv.compute kse
        dKse |> Deriv.ofVar lTensor
    )




[<Fact>]
let ``Eval and derive: lkse`` () =
    //input     mu[gp]
    //          sigma[gp,gp]
    //          x[gp,smpl]
    //          l[gp]
    //output    lk[ga, smpl]

    printfn "======= Testing lk:"

    let nGps = SizeSpec.symbol "nGps"
    let nSmpls = SizeSpec.symbol "nSmpls"
    let gp = ElemExpr.idx 0
    let smpl = ElemExpr.idx 1

    let mu = ElemExpr.argElem 0
    let sigma = ElemExpr.argElem 1
    let x = ElemExpr.argElem 2
    let l = ElemExpr.argElem 3
    let lkse = sqrt( l[gp]**2.0 / (l[gp]**2.0 + sigma[gp;gp]) ) * exp(- ((mu[gp]- x[gp;smpl])**2.0) / (2.0 * (l[gp]**2.0 + sigma[gp;gp])) )
    let dLkse = ElemExprDeriv.buildDerivElemExpr lkse [nGps;nSmpls] 4

    printfn "lk=\n%A" lkse
    printfn "dlk / dmu=\n%A" dLkse.[0]
    printfn "dlk / dSigma=\n%A" dLkse.[1]
    printfn "dlk / dx=\n%A" dLkse.[2]
    printfn "dlk / dl=\n%A" dLkse.[3]

    let xVal = [[1.0; 1.1; 2.0];[1.0; 1.1; 2.0]] |> ArrayNDHost.ofList2D
    let lVal = [0.5;0.6] |> ArrayNDHost.ofList
    let muVal = [1.0;0.5] |> ArrayNDHost.ofList
    let sigmaVal = [[0.4;0.2];[0.2;0.8]] |> ArrayNDHost.ofList2D
    let lkseVal = ElemExpr.eval lkse [muVal;sigmaVal;xVal;lVal] [2;3]

    printfn "mu=\n%A" muVal
    printfn "sigma=\n%A" sigmaVal
    printfn "x=\n%A" xVal
    printfn "l=\n%A" lVal
    printfn "lkse=\n%A" lkseVal

    let dlk0Val = ElemExpr.eval dLkse.[0] [muVal;sigmaVal;xVal;lVal;lkseVal] [2]
    let dlk1Val = ElemExpr.eval dLkse.[1] [muVal;sigmaVal;xVal;lVal;lkseVal] [2;2]
    let dlk2Val = ElemExpr.eval dLkse.[2] [muVal;sigmaVal;xVal;lVal;lkseVal] [2;3]
    let dlk3Val = ElemExpr.eval dLkse.[3] [muVal;sigmaVal;xVal;lVal;lkseVal] [2]
    printfn "dlkse / dmu=\n%A" dlk0Val
    printfn "dlkse / dsigma=\n%A" dlk1Val
    printfn "dlkse / dx=\n%A" dlk2Val
    printfn "dlkse / dl=\n%A" dlk3Val