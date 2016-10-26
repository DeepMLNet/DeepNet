namespace ModelTests
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open System

module GPTests=


    let EPTest()  =
        let rand = Random(1)

        let nX = 5
        let nXm = 6
        let nIn = 10


        let n = SizeSpec.symbol "XSize"
        let m = SizeSpec.symbol "XmSize"
        let i = SizeSpec.symbol "XstarSize"
        let x = Expr.var<single> "X" [n]
        let y = Expr.var<single> "Y" [n]
        let xm = Expr.var<single> "Xm" [m]
        let xStar = Expr.var<single> "Xstar" [i]
        let l = Expr.var<single> "l" []
        let sigN = Expr.var<single> "SigmaNs" [n]
        let sigf = Expr.var<single> "Sigf" []
        let vu = 1e-6f
        
        let covPdPd  (x:ExprT) (y:ExprT) (k:ExprT)  = 
            let i, j  = ElemExpr.idx2
            let cMat,xvect,yvect = ElemExpr.arg3<single>
            let xelem = xvect[i]
            let yelem = yvect[j]
            let cPdPd = cMat[i;j] |> ElemExprDeriv.compute  |> ElemExprDeriv.ofArgElem xelem |> ElemExprDeriv.compute  |> ElemExprDeriv.ofArgElem yelem
            let sizeK1 = k.Shape.[0]
            let sizeK2 = k.Shape.[1]
            Expr.elements [sizeK1;sizeK2] cPdPd [k;x;y]
        
        let covPdFun (x:ExprT) (k:ExprT) =
            let i, j  = ElemExpr.idx2
            let cMat,xvect = ElemExpr.arg2<single>
            let xelem = xvect[i]
            let cPdFun = cMat[i;j] |> ElemExprDeriv.compute  |> ElemExprDeriv.ofArgElem xelem
            let sizeK1 = k.Shape.[0]
            let sizeK2 = k.Shape.[1]
            Expr.elements [sizeK1;sizeK2] cPdFun [k;x]

        let kFf = GaussianProcess.squaredExpCovariance (l,sigf) x x
        let kmm = GaussianProcess.squaredExpCovariance (l,sigf) xm xm
        let covSite, muSite = ExpectationPropagation.monotonicityEP kmm vu 10
        let kF'f = GaussianProcess.covPdFunSE (l,sigf) xm x 
//        let kF'f = GaussianProcess.squaredExpCovariance (l,sigf) xm x |> covPdFun xm
        let kFf' = kF'f.T
        let kF'f' = GaussianProcess.covPdPdSE (l,sigf)  xm xm 
//        let kF'f' = GaussianProcess.squaredExpCovariance (l,sigf) xm xm |> covPdPd xm xm
        let xJoint = Expr.concat 0 [x;xm]
        let kJoint1,kJoint2 = Expr.concat 1 [kFf;kFf'],Expr.concat 1 [kF'f;kF'f']
        let kJoint = Expr.concat 0 [kJoint1;kJoint2]
        let muJoint = Expr.concat 0 [y;muSite]
        let sigmaJoint =  Expr.concat 0 [sigN; covSite] |> Expr.diagMat
        let kInv = Expr.invert (kJoint + sigmaJoint)
        let kStar = GaussianProcess.squaredExpCovariance (l,sigf) xJoint xStar
        let kStarStar = GaussianProcess.squaredExpCovariance (l,sigf) xStar xStar
        let mean =  kStar.T .* kInv .* (muJoint)
        let cov = kStarStar - kStar.T .* kInv .* kStar

       

        let cmplr = DevCuda.Compiler, CompileEnv.empty
        let covSiteFun  = Func.make<single> cmplr covSite |> arg3 xm l sigf
        let muSiteFun   = Func.make<single> cmplr muSite |> arg3 xm l sigf
        
        let kFun = Func.make<single> cmplr kFf |> arg3 x l sigf
        let kJointFun  = Func.make<single> cmplr kJoint|> arg4 xm x l sigf
        let sigmaJointFun = Func.make<single> cmplr sigmaJoint|> arg4 xm l sigf sigN
        let muJointFun = Func.make<single> cmplr muJoint |> arg4 xm y l sigf
        let kStarFun = Func.make<single> cmplr kStar |> arg5 xm x xStar l sigf
        let kStarStarFun = Func.make<single> cmplr kStarStar |> arg3 xStar l sigf

        let meanFun = Func.make<single> cmplr mean |> arg7 xm x xStar y l sigf sigN
        let covFun = Func.make<single> cmplr cov |> arg7 xm x xStar y l sigf sigN

        let lValue = ArrayNDHost.scalar 1.0f |> ArrayNDCuda.toDev
        let sigfValue = ArrayNDHost.scalar 1.0f |> ArrayNDCuda.toDev
        let xValue = rand.SortedUniformArrayND (-2.0f, 2.0f) [nX] |> ArrayNDCuda.toDev 
        let xmValue = ArrayNDHost.linSpaced -2.0f 2.0f nXm |> ArrayNDCuda.toDev
        let yValue = ArrayND.map (fun x -> 0.5f * x ** 3.0f - x) xValue
        let xStarValue = rand.UniformArrayND (-2.0f, 2.0f) [nIn] |> ArrayNDCuda.toDev
        let sigNValue = (ArrayND.onesLike xValue) * sqrt(0.1f)

        printfn "x = \n %A" xValue
        printfn "y = \n %A" yValue
        printfn "xm = \n %A" xmValue
        printfn "xStar = \n %A" xStarValue


        let covSite = covSiteFun xmValue lValue sigfValue
        let muSite = muSiteFun xmValue lValue sigfValue
        printfn "covSite = \n%A" covSite
        printfn "muSite = \n%A" muSite

        let k = kFun xValue lValue sigfValue 
        printfn "k = \n%A" k

        let kJoint = kJointFun xmValue xValue lValue sigfValue
        let sigmaJoint = sigmaJointFun xmValue lValue sigfValue sigNValue
        let muJoint = muJointFun xmValue yValue lValue sigfValue
        printfn "kJoint = \n%A" kJoint
        printfn "sigmaJoint = \n%A" sigmaJoint
        printfn "muJoint = \n%A" muJoint
        printfn "muJointLen = %i" muJoint.NElems
        let kStar = kStarFun xmValue xValue xStarValue lValue sigfValue
        let kStarStar =  kStarStarFun xStarValue  lValue sigfValue
        printfn "kStar = \n%A" kStar
        printfn "kStarStar = \n%A" kStarStar 

        printfn "kJoint + SigmaJoint  = \n%A" (kJoint + sigmaJoint)

        let kInv = ArrayND.invert (kJoint + sigmaJoint)
        printfn "kInv = \n%A" kInv

        let interim = kStar.T .* kInv .* kStar
        printfn "kStar.T .* kInv .* kStar = \n%A" interim

        let mean = meanFun xmValue xValue xStarValue yValue lValue sigfValue sigNValue
        let cov = covFun xmValue xValue xStarValue yValue lValue sigfValue sigNValue
        printfn "mean = \n%A" mean
        printfn "cov = \n%A" cov