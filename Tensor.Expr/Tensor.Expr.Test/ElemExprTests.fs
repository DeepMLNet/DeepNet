namespace global
#nowarn "25"

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor
open Tensor.Expr
open TestUtils


type ElemExprTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 
  
    [<Fact>]
    let ``Eval: simple`` () =   
        // input  x[i, j]
        //        y[m, n]
        // output  [k]

        printfn "======= Testing evaluation:"

        let k = Elem.Expr.idx 0   
        let x = Elem.Expr.argElem<float> 0
        let y = Elem.Expr.argElem<float> 1
        let expr = 2.0 * (x [k; k]) + y [Size.zero; k]

        let xVal = [1.0; 2.0; 3.0] |> HostTensor.ofList |> Tensor.diagMat
        let yVal = [[4.0; 5.0; 6.0]
                    [7.0; 8.0; 9.0]] |> HostTensor.ofList2D
        let res = Elem.ExprHostEval.eval expr [xVal; yVal] [xVal.Shape.[0]]

        printfn "Expr:\n%A" expr
        printfn "x=\n%A" xVal
        printfn "y=\n%A" yVal
        printfn "result=\n%A" res

        let expected = [6.0; 9.0; 12.0] |> HostTensor.ofList
        Tensor.almostEqual res expected |> should equal true


    [<Fact>]
    let ``Eval: sum`` () =   
        // input  x[i, j]
        // output  [k]

        printfn "======= Testing evaluation sum:"

        let is = Size.fix 3L
        let js = Size.fix 3L

        let k = Elem.Expr.idx 0   
        let x = Elem.Expr.argElem<float> 0
        let l = Elem.Expr.sumIdx "l"
        let expr = 2.0 * (Elem.Expr.sum l Size.zero (is-1L) (x [l; k]))

        let xVal = [1.0; 2.0; 3.0] |> HostTensor.ofList |> Tensor.diagMat
        let xVal = xVal + HostTensor.ones [3L; 3L]
        let res = Elem.ExprHostEval.eval expr [xVal] [xVal.Shape.[0]]

        printfn "Expr:\n%A" expr
        printfn "x=\n%A" xVal
        printfn "result=\n%A" res

        let expected = [8.0; 10.0; 12.0] |> HostTensor.ofList
        Tensor.almostEqual res expected  |> should equal true

    [<Fact>]
    let ``Deriv: 1`` () =   
        // input  x[i, j]
        //        y[m, n]
        // output  [k]

        printfn "======= Testing derivatives:"

        let ks = Size.symbol "ks"

        let k = Elem.Expr.idx 0   
        let x = Elem.Expr.argElem<float> 0
        let y = Elem.Expr.argElem<float> 1
        let expr = 2.0 * (x [k; k]) + y [Size.zero; k]
        let dExpr = Elem.ExprDeriv.buildDerivElem.Expr expr [ks] 2

        printfn "Expr:\n%A" expr
        printfn "dExpr / dx:\n%A" dExpr.[0]
        printfn "dExpr / dy:\n%A" dExpr.[1]


    [<Fact>]
    let ``Deriv: 2`` () =   
        // input  x[i, j]
        //        y[m, n]
        // output  [k, l]

        printfn "======= Testing derivatives 2:"

        let ks = Size.symbol "ks"
        let ls = Size.symbol "ls"
        let k = Elem.Expr.idx 0   
        let l = Elem.Expr.idx 1

        let x = Elem.Expr.argElem<float> 0
        let y = Elem.Expr.argElem<float> 1
        let expr = 2.0 * (x [k; k]) + y [l; k]
        let dExpr = Elem.ExprDeriv.buildDerivElem.Expr expr [ks; ls] 2

        printfn "Expr:\n%A" expr
        printfn "dExpr / dx:\n%A" dExpr.[0]
        printfn "dExpr / dy:\n%A" dExpr.[1]


    [<Fact>]
    let ``Deriv: sum`` () =   
        // input  x[i, j]
        // output  [k]

        printfn "======= Testing derivative sum:"

        let is = Size.fix 3L
        let js = Size.fix 3L

        let k = Elem.Expr.idx 0   
        let x = Elem.Expr.argElem<float> 0
        let l = Elem.Expr.sumIdx "l"
        let expr = 2.0 * (Elem.Expr.sum l Size.zero (is-1L) (x [l; k]))
        let dExpr = Elem.ExprDeriv.buildDerivElem.Expr expr [is] 1

        printfn "Expr:\n%A" expr
        printfn "dExpr/dx:\n%A" dExpr.[0]
   

    [<Fact>]
    let ``Eval and deriv: KSE`` () =   
        // input  x[gp, smpl]
        //        l[gp]
        // output cov[gp, smpl1, smpl2]

        printfn "======= Testing KSE:"

        let nGps = Size.fix 1L
        let nSmpls = Size.fix 3L
        let gp = Elem.Expr.idx 0   
        let smpl1 = Elem.Expr.idx 1
        let smpl2 = Elem.Expr.idx 2

        let x = Elem.Expr.argElem<float> 0
        let l = Elem.Expr.argElem<float> 1    
        let kse = exp (- ((x [gp; smpl1] - x [gp; smpl2])***2.0) / (2.0 * (l [gp])***2.0) )
        let dKse = Elem.ExprDeriv.buildDerivElem.Expr kse [nGps; nSmpls; nSmpls] 2

        printfn "KSE:\n%A" kse
        printfn "dKSE / dx:\n%A" dKse.[0]
        printfn "dKSE / dl:\n%A" dKse.[1]


        let xVal = [[1.0; 1.1; 2.0]] |> HostTensor.ofList2D
        let lVal = [0.5] |> HostTensor.ofList
        let kseVal = Elem.ExprHostEval.eval kse [xVal; lVal] [1L; 3L; 3L]

        printfn "x=\n%A" xVal
        printfn "l=\n%A" lVal
        printfn "kse=\n%A" kseVal

        let dKseVal = kseVal |> Tensor.reshape [1L; 1L; 3L; 3L]

        let dKSe0Val = Elem.ExprHostEval.eval dKse.[0] [xVal; lVal; dKseVal] [1L; 1L; 3L]
        let dKSe1Val = Elem.ExprHostEval.eval dKse.[1] [xVal; lVal; dKseVal] [1L; 1L]
        printfn "dkse / dx=\n%A" dKSe0Val
        printfn "dkse / dl=\n%A" dKSe1Val


    [<Fact>]
    let ``Codegen: KSE`` () =  
        printfn "======= Testing KSE codegen:"

        let nGps = Size.fix 1L
        let nSmpls = Size.fix 3L
        let gp = Elem.Expr.idx 0   
        let smpl1 = Elem.Expr.idx 1
        let smpl2 = Elem.Expr.idx 2

        let x = Elem.Expr.argElem<float> 0
        let l = Elem.Expr.argElem<float> 1
        let kse = exp (- ((x [gp; smpl1] - x [gp; smpl2])***2.0) / (2.0 * (l [gp])***2.0) )
        let dKse = Elem.ExprDeriv.buildDerivElem.Expr kse [nGps; nSmpls; nSmpls] 2
        let dKsedX, dKsedL = dKse.[0], dKse.[1]

        let uKse = UElem.Expr.toUElemFunc kse 3 2 
        let udKsedX = UElem.Expr.toUElemFunc dKsedX (2+1) (2+1) 
        let udKsedL = UElem.Expr.toUElemFunc dKsedL (1+1) (2+1) 
        let kseCode = CudaElem.Expr.generateFunctor "KSE" uKse (Permutation.identity uKse.NDims)
        let dKsedXCode = CudaElem.Expr.generateFunctor "dKSEdX" udKsedX (Permutation.identity udKsedX.NDims)
        let dKsedLCode = CudaElem.Expr.generateFunctor "dKSEdL" udKsedL (Permutation.identity udKsedL.NDims)

        printfn "Code:\n%s%s%s" kseCode dKsedXCode dKsedLCode


    [<Fact>]
    let ``Eval and deriv: KSE in Expr on Host`` () =   
        // input  x[gp, smpl]
        //        l[gp] 
        // output cov[gp, smpl1, smpl2]

        printfn "======= Testing KSE in Expr on Host:"

        let nGps = Size.symbol "nGps"
        let nSmpls = Size.symbol "nSmpls"
        let gp = Elem.Expr.idx 0   
        let smpl1 = Elem.Expr.idx 1
        let smpl2 = Elem.Expr.idx 2

        let x = Elem.Expr.argElem<float> 0
        let l = Elem.Expr.argElem<float> 1
        let kseExpr = exp (- ((x [gp; smpl1] - x [gp; smpl2])***2.0) / (2.0 * (l [gp])***2.0) )

        let xTensor = Expr.var<double> "xTensor" [nGps; nSmpls] 
        let lTensor = Expr.var<double> "lTensor" [nGps]
        let kse = Expr.elements [nGps; nSmpls; nSmpls] kseExpr [xTensor; lTensor]

        let dKse = Deriv.compute kse
        let dKsedX = dKse |> Deriv.ofVar xTensor
        let dKsedL = dKse |> Deriv.ofVar lTensor

        let kseFn = Func.make<float> DevHost.DefaultFactory kse |> arg2 xTensor lTensor
        let dKseFn = Func.make2<float, float> DevHost.DefaultFactory dKsedX dKsedL |> arg2 xTensor lTensor

        let kseinv = Expr.invert kse  
    
        let dKseinv = Deriv.compute kseinv
        let dKseinvdX  = dKseinv |> Deriv.ofVar xTensor
        let dKseinvdL  = dKseinv |> Deriv.ofVar lTensor

        let kseinvFn = Func.make<float> DevHost.DefaultFactory kseinv |> arg2 xTensor lTensor
        let dKseinvFn = Func.make2<float, float> DevHost.DefaultFactory dKseinvdX dKseinvdL |> arg2 xTensor lTensor

        let xVal = [[1.0; 1.1; 2.0]] |> HostTensor.ofList2D
        let lVal = [0.5] |> HostTensor.ofList

        let kseVal = kseFn xVal lVal
        let dKsedXVal, dKsedLVal = dKseFn xVal lVal

        let kseinvVal = kseinvFn xVal lVal
        let dKseinvdXVal, dKseinvdLVal = dKseinvFn xVal lVal

        printfn "x=\n%A" xVal
        printfn "l=\n%A" lVal
        printfn "kse=\n%A" kseVal
        printfn "dkse / dx=\n%A" dKsedXVal
        printfn "dkse / dl=\n%A" dKsedLVal
        printfn "kseinv=\n%A" kseinvVal
        printfn "dkseinv / dx=\n%A" dKseinvdXVal
        printfn "dkseinv / dl=\n%A" dKseinvdLVal

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Eval and deriv: KSE in Expr on CUDA`` () =   
        // input  x[gp, smpl]
        //        l[gp]
        // output cov[gp, smpl1, smpl2]

        printfn "======= Testing KSE in Expr on CUDA:"

        let nGps = Size.symbol "nGps"
        let nSmpls = Size.symbol "nSmpls"
        let gp = Elem.Expr.idx 0   
        let smpl1 = Elem.Expr.idx 1
        let smpl2 = Elem.Expr.idx 2

        let x = Elem.Expr.argElem<single> 0
        let l = Elem.Expr.argElem<single> 1
        let kseExpr = exp (- ((x [gp; smpl1] - x [gp; smpl2])***2.0f) / (2.0f * (l [gp])***2.0f) )

        let xTensor = Expr.var<single> "xTensor" [nGps; nSmpls] 
        let lTensor = Expr.var<single> "lTensor" [nGps]
        let kse = Expr.elements [nGps; nSmpls; nSmpls] kseExpr [xTensor; lTensor]

        let dKse = Deriv.compute kse
        let dKsedX = dKse |> Deriv.ofVar xTensor
        let dKsedL = dKse |> Deriv.ofVar lTensor

        let kseFn = Func.make<single> DevCuda.DefaultFactory kse |> arg2 xTensor lTensor
        let dKseFn = Func.make2<single, single> DevCuda.DefaultFactory dKsedX dKsedL |> arg2 xTensor lTensor

        let xVal = [[1.0f; 1.1f; 2.0f]] |> HostTensor.ofList2D |> CudaTensor.transfer
        let lVal = [0.5f] |> HostTensor.ofList |> CudaTensor.transfer

        let kseVal = kseFn xVal lVal
        let dKsedXVal, dKsedLVal = dKseFn xVal lVal

        printfn "x=\n%A" xVal
        printfn "l=\n%A" lVal
        printfn "kse=\n%A" kseVal
        printfn "dkse / dx=\n%A" dKsedXVal
        printfn "dkse / dl=\n%A" dKsedLVal


    let kseElem.Expr () = 
        // input  x[gp, smpl]
        //        l[gp]
        // output cov[gp, smpl1, smpl2]
        let gp = Elem.Expr.idx 0   
        let smpl1 = Elem.Expr.idx 1
        let smpl2 = Elem.Expr.idx 2
        let x = Elem.Expr.argElem<single> 0
        let l = Elem.Expr.argElem<single> 1
        exp (- ((x [gp; smpl1] - x [gp; smpl2])***2.0f) / (2.0f * (l [gp])***2.0f) )

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Trace compare: KSE`` () =    
        let nGps = 2L
        let nSmpls = 3L
        requireEqualTracesWithRandomData [[nGps; nSmpls]; [nGps]] (fun [xTensor; lTensor] ->
            let Elem.Expr = kseElem.Expr ()
            let kse = 
                Expr.elements [Size.fix nGps; Size.fix nSmpls; Size.fix nSmpls] 
                    Elem.Expr [xTensor; lTensor]
            kse    
        )

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Trace compare: dKSE/dX`` () =    
        let nGps = 2L
        let nSmpls = 3L
        requireEqualTracesWithRandomData [[nGps; nSmpls]; [nGps]] (fun [xTensor; lTensor] ->
            let Elem.Expr = kseElem.Expr ()
            let kse = 
                Expr.elements [Size.fix nGps; Size.fix nSmpls; Size.fix nSmpls] 
                    Elem.Expr [xTensor; lTensor]
            let dKse = Deriv.compute kse
            dKse |> Deriv.ofVar xTensor
        )

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Trace compare: dKSE/dL`` () =    
        let nGps = 2L
        let nSmpls = 3L
        requireEqualTracesWithRandomData [[nGps; nSmpls]; [nGps]] (fun [xTensor; lTensor] ->
            let Elem.Expr = kseElem.Expr ()
            let kse = 
                Expr.elements [Size.fix nGps; Size.fix nSmpls; Size.fix nSmpls] 
                    Elem.Expr [xTensor; lTensor]
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

        let nGps = Size.fix 2L
        let nSmpls = Size.fix 3L
        let gp = Elem.Expr.idx 0
        let smpl = Elem.Expr.idx 1

        let mu = Elem.Expr.argElem<float> 0
        let sigma = Elem.Expr.argElem<float> 1
        let x = Elem.Expr.argElem<float> 2
        let l = Elem.Expr.argElem<float> 3
        let lkse = sqrt( l[gp]***2.0 / (l[gp]***2.0 + sigma[gp;gp]) ) * exp(- ((mu[gp]- x[gp;smpl])***2.0) / (2.0 * (l[gp]***2.0 + sigma[gp;gp])) )
        let dLkse = Elem.ExprDeriv.buildDerivElem.Expr lkse [nGps;nSmpls] 4

        printfn "lk=\n%A" lkse
        printfn "dlk / dmu=\n%A" dLkse.[0]
        printfn "dlk / dSigma=\n%A" dLkse.[1]
        printfn "dlk / dx=\n%A" dLkse.[2]
        printfn "dlk / dl=\n%A" dLkse.[3]

        let xVal = [[1.0; 1.1; 2.0];[1.0; 1.1; 2.0]] |> HostTensor.ofList2D
        let lVal = [0.5;0.6] |> HostTensor.ofList
        let muVal = [1.0;0.5] |> HostTensor.ofList
        let sigmaVal = [[0.4;0.2];[0.2;0.8]] |> HostTensor.ofList2D
        let lkseVal = Elem.ExprHostEval.eval lkse [muVal;sigmaVal;xVal;lVal] [2L; 3L]

        printfn "mu=\n%A" muVal
        printfn "sigma=\n%A" sigmaVal
        printfn "x=\n%A" xVal
        printfn "l=\n%A" lVal
        printfn "lkse=\n%A" lkseVal

        let dlkseVal = lkseVal |> Tensor.reshape [1L; 2L; 3L]

        let dlk0Val = Elem.ExprHostEval.eval dLkse.[0] [muVal;sigmaVal;xVal;lVal;dlkseVal] [1L; 2L]
        let dlk1Val = Elem.ExprHostEval.eval dLkse.[1] [muVal;sigmaVal;xVal;lVal;dlkseVal] [1L; 2L; 2L]
        let dlk2Val = Elem.ExprHostEval.eval dLkse.[2] [muVal;sigmaVal;xVal;lVal;dlkseVal] [1L; 2L; 3L]
        let dlk3Val = Elem.ExprHostEval.eval dLkse.[3] [muVal;sigmaVal;xVal;lVal;dlkseVal] [1L; 2L]
        printfn "dlkse / dmu=\n%A" dlk0Val
        printfn "dlkse / dsigma=\n%A" dlk1Val
        printfn "dlkse / dx=\n%A" dlk2Val
        printfn "dlkse / dl=\n%A" dlk3Val


    [<Fact>]
    let ``DerivTest: GP Predict`` () =   
        /// squared exponential covariance matrix
        let sqExpCov (l:ExprT, sigf:ExprT) x y =
            let x_smpl, y_smpl = Elem.Expr.idx2
            let xvec, yvec,len,sigmaf = Elem.Expr.arg4<double>
            let kse = sigmaf[] * exp  (-( (xvec[x_smpl] - yvec[y_smpl]) *** 2.0) / (2.0 * len[] *** 2.0) )
            let sizeX = Expr.nElems x
            let sizeY = Expr.nElems y
            Expr.elements [sizeX; sizeY] kse [x; y; l; sigf]

        // variables
        let nTrnSmpls = Size.fix 5L
        let nTstSmpls = Size.fix 4L
        let l = Expr.var<double> "l" []
        let sigf = Expr.var<double> "sigf" []
        let x = Expr.var<double> "x" [nTrnSmpls]
        let t = Expr.var<double> "t" [nTrnSmpls]
        let sign = Expr.var<double> "sign" [nTrnSmpls]
        let x' = Expr.var<double> "x'" [nTstSmpls]

        // GP prediction
        let k  = sqExpCov (l, sigf) x x + Expr.diagMat sign
        let k' = sqExpCov (l, sigf) x' x
        let mean = k' .* (Expr.invert k) .* t

        // do check
        let xv = HostTensor.linspace -4.0 4.0 5L
        let x'v = HostTensor.linspace -3.0 3.0 4L
        let varEnv = VarEnv.ofSeq [l, HostTensor.scalar 1.0
                                   sigf, HostTensor.scalar 1.0
                                   x, xv
                                   t, xv |> Tensor.map tanh
                                   sign, 0.001 * Tensor.onesLike xv
                                   x', x'v
                                  ]
        DerivCheck.checkExprTree DevHost 1e-6 1e-7 varEnv mean


    [<Fact>]
    let ``DerivTest: GP Predict2`` () =   
        /// squared exponential covariance matrix
        let sqExpCov (l:ExprT, sigf:ExprT) x y =
            let x_smpl, y_smpl = Elem.Expr.idx2
            let xvec, yvec,len,sigmaf = Elem.Expr.arg4<double>
            //let kse = sigmaf[] * exp  (-( (xvec[x_smpl] - yvec[y_smpl]) *** 2.0) / (2.0 * len[] *** 2.0) )
            //let kse = exp  (-( (xvec[x_smpl] - yvec[y_smpl]) ) )
            let kse = xvec[x_smpl] + yvec[y_smpl]
            let sizeX = Expr.nElems x
            let sizeY = Expr.nElems y
            Expr.elements [sizeX; sizeY] kse [x; y; l; sigf]

        // variables
        let nTrnSmpls = Size.fix 15L
        let nTstSmpls = Size.fix 10L
        let l = Expr.scalar 1.0
        let sigf = Expr.scalar 1.0
        let x = Expr.var<double> "x" [nTrnSmpls]
        let t = Expr.var<double> "t" [nTrnSmpls]
        let sign = Expr.var<double> "sign" [nTrnSmpls]
        let x' = Expr.var<double> "x'" [nTstSmpls]

        // GP prediction
        //let k  = sqExpCov (l, sigf) x x + Expr.diagMat sign
        //let k  = Expr.identity<double> nTrnSmpls
        let k' = sqExpCov (l, sigf) x' x
        let mean = k' //.* t
        //let mean = k' .* (Expr.invert k) .* t

        let dmean = mean |> Deriv.compute |> Deriv.ofVar x
        printfn "%A" dmean

        // do check
        let xv = HostTensor.linspace -4.0 4.0 15L
        let x'v = HostTensor.linspace -3.0 3.0 10L
        let varEnv = VarEnv.ofSeq [                               
                                   x, xv
                                   t, xv |> Tensor.map tanh
                                   sign, 0.001 * Tensor.onesLike xv
                                   x', x'v
                                  ]
        //SymTensor.Debug.VisualizeUExpr <- true
        //SymTensor.Debug.DisableOptimizer <- true 

        DerivCheck.checkExprTree DevHost 1e-6 1e-7 varEnv mean

