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
        let ninput = 20

        let trn_x_list =  (TestFunctions.randomSortedListOfLength rand (-5.0f,-1.0f) (ntraining/2)) @  (TestFunctions.randomSortedListOfLength rand (1.0f,5.0f) (ntraining/2))
        let trn_x_host = trn_x_list |> ArrayNDHost.ofList
        let trn_t_list = trn_x_list |>  TestFunctions.randPolynomial rand
        let trn_t_host = trn_t_list |> ArrayNDHost.ofList

        let sigmaNs_host = (ArrayNDHost.ones<single> [ntraining]) * sqrt 0.001f

        //transfer train parametters to device (Host or GPU)
        let trn_x_val = trn_x_host  |> TestFunctions.post DevCuda
        let trn_t_val = trn_t_host  |> TestFunctions.post DevCuda
        let sigmaNs_val = sigmaNs_host  |> TestFunctions.post DevCuda

        printfn "Trn_x =\n%A" trn_x_host
        printfn "Trn_t =\n%A" trn_t_host
        let kernel = GaussianProcess.SquaredExponential (1.0f,1.0f)
        let hyperPars = {GaussianProcess.Kernel =kernel}
        let range = (-0.5f,0.5f)
        let smpls, mean_smpls, cov_smpls, stdev_smpls = GPPlots.predictGP hyperPars sigmaNs_val trn_x_val trn_t_val range ninput
        printfn "Sample points =\n%A" smpls
        printfn "Sampled means =\n%A" mean_smpls
        printfn "Sampled Covariances =\n%A" cov_smpls
        printfn "Sampled StanderdDeviations =\n%A" stdev_smpls
        let gpTestPlot = fun () -> GPPlots.simplePlot (hyperPars, sigmaNs_val, trn_x_val, trn_t_val)
        gpTestPlot ()
        save "GPTestplot.png" gpTestPlot
        ()
    ()


