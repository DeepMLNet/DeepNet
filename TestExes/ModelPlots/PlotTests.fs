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
open RTools
open RProvider
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

        //transfer train parametters to device (Host or GPU)
        let trnXVal = trnXHost  |> TestFunctions.post DevCuda
        let trnTVal = trnTHost  |> TestFunctions.post DevCuda
        let sigmaNsVal = sigmaNs_host  |> TestFunctions.post DevCuda
        let trn_x_val2 = ArrayND.copy trnXVal
        let trn_t_val2 = ArrayND.copy trnTVal
        let sigmaNs_val2 = sigmaNsVal
        printfn "Trn_x =\n%A" trnXHost
        printfn "Trn_t =\n%A" trnTHost
        let zeroMean (x:ExprT) = Expr.zerosLike x
        let tanHMean (x:ExprT) = tanh x
        let hyperPars = {GaussianProcess.Kernel =GaussianProcess.SquaredExponential (1.0f,1.0f)
                         GaussianProcess.MeanFunction = tanHMean
                         GaussianProcess.Monotonicity = None
                         GaussianProcess.CutOutsideRange = false}
        let range = (-0.5f,0.5f)
        let smpls, mean_smpls, cov_smpls, stdev_smpls = GPPlots.Plots.predictGP hyperPars sigmaNsVal trnXVal trnTVal range ninput
        printfn "Sample points =\n%A" smpls
        printfn "Sampled means =\n%A" mean_smpls
        printfn "Sampled Covariances =\n%A" cov_smpls
        printfn "Sampled StanderdDeviations =\n%A" stdev_smpls
        let gpTestPlot = fun () -> GPPlots.Plots.simplePlot (hyperPars, sigmaNsVal, trnXVal, trnTVal,ninput)
        gpTestPlot ()
        save "GPTestplot1.png" gpTestPlot

    ()

    let multiplotTest () =
        let x = [-5.0 .. 5.0]
        let y = List.map (fun x -> x**2.0) x
        let negx = List.map (fun x -> -x) x
        let negy = List.map (fun x -> -x) y
        let plots = ["0deg", fun () -> namedParams[
                                            "x", box x
                                            "y", box y]
                                            |> R.plot
                                            |> ignore
                     "90deg", fun () -> namedParams[
                                            "x", box y
                                            "y", box x]
                                            |> R.plot
                                            |> ignore
                     "270deg", fun () ->namedParams[
                                            "x", box negy
                                            "y", box negx]
                                            |> R.plot
                                            |> ignore
                     "180deg", fun () -> namedParams[
                                            "x", box negx
                                            "y", box negy]
                                            |> R.plot
                                            |> ignore
                    ]
        namedParams [
            "mfrow", box [2;2]]
        |> R.par |> ignore
        plots |> List.map (fun (title,plot)-> 
                plot ()
                namedParams [
                    "main", box title]
                |>R.title|> ignore
                ) |> ignore