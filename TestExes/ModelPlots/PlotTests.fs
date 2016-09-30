namespace ModelPlots
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open RProvider
open RProvider.ggplot2
open Basics
open GPAct
open System
open Models
open RProvider.graphics
open MLPlots
module PlotTests =
 
    let save = savePlot 400 600 Environment.CurrentDirectory

    let GPTransferTest () =
        
        let seed = 1
        let rand = Random seed
        let ntraining = 100
        let ninput = 5000

        let trnXList =  (TestFunctions.randomSortedListOfLength rand (-5.0f,-1.0f) (ntraining/2)) @  (TestFunctions.randomSortedListOfLength rand (1.0f,5.0f) (ntraining/2))
        let trnXHost = trnXList |> ArrayNDHost.ofList
        let trnTList = trnXList |>  TestFunctions.randPolynomial rand
        let trnTHost = trnTList |> ArrayNDHost.ofList

        let sigmaNs_host = (ArrayNDHost.ones<single> [ntraining]) * sqrt 0.001f

        let newArray (ary:ArrayNDT<single>)=
            let aNew = ArrayND.newCOfType  ary.Shape ary
            ArrayND.copyTo ary aNew
            aNew
        //transfer train parametters to device (Host or GPU)
        let trnXVal = trnXHost  |> TestFunctions.post DevCuda
        let trnTVal = trnTHost  |> TestFunctions.post DevCuda
        let sigmaNsVal = sigmaNs_host  |> TestFunctions.post DevCuda
        let trn_x_val2 = newArray trnXVal
        let trn_t_val2 = newArray trnTVal
        let sigmaNs_val2 = sigmaNsVal
        printfn "Trn_x =\n%A" trnXHost
        printfn "Trn_t =\n%A" trnTHost
        let hyperPars = {GaussianProcess.Kernel =GaussianProcess.SquaredExponential (1.0f,1.0f)}
        let range = (-0.5f,0.5f)
        let smpls, mean_smpls, cov_smpls, stdev_smpls = GPPlots.predictGP hyperPars sigmaNsVal trnXVal trnTVal range ninput
        printfn "Sample points =\n%A" smpls
        printfn "Sampled means =\n%A" mean_smpls
        printfn "Sampled Covariances =\n%A" cov_smpls
        printfn "Sampled StanderdDeviations =\n%A" stdev_smpls
        let gpTestPlot = fun () -> GPPlots.simplePlot (hyperPars, sigmaNsVal, trnXVal, trnTVal,ninput)
        gpTestPlot ()
        save "GPTestplot1.png" gpTestPlot

    ()


