namespace ModelTests
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open System

module GPTests=


    let EPTest()  =
        let rand = Random(1)
        let n = SizeSpec.symbol "n"
        let k = Expr.var<single> "K" [n;n]
        let vu = 1e-6f
        let covSite, muSite = ExpectationPropagation.monotonicityEP k vu 10
        let cmplr = DevCuda.Compiler, CompileEnv.empty
        let covSiteFun : ArrayNDT<single> -> ArrayNDT<single> = Func.make cmplr covSite |> arg1 k
        let muSiteFun : ArrayNDT<single> -> ArrayNDT<single> = Func.make cmplr muSite |> arg1 k
        let kValue = rand.UniformArrayND (0.0f, 2.0f) [10;10] 
        let kValue = kValue .* kValue |> ArrayNDCuda.toDev
        let covSite = covSiteFun kValue
        let muSite = muSiteFun kValue
        printfn "K = \n%A" kValue
        printfn "covSite = \n%A" covSite
        printfn "muSite = \n%A" muSite