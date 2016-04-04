module Controller

open System

open Basics
open ArrayNDNS
open Datasets
open SymTensor
open SymTensor.Compiler.Cuda
open Optimizers
open Models
open Data


type IController =

    /// Compute control signal given Biotac sensor data.
    /// Input:  Biotac.[smpl, chnl]
    /// Output: Velocity.[smpl, dim]
    abstract Predict: biotac: Arrays -> Arrays


type MLPControllerCfg = {
    MLP:            MLP.HyperPars
    BatchSize:      int
    Seed:           int
    Iters:          int
    StepSize:       single
}

let nBiotac = SizeSpec.symbol "nBiotac"
let nVelocity = SizeSpec.symbol "nVelocity"

type MLPController (cfg:   MLPControllerCfg) =

    let mc = ModelBuilder<single> "MLPController"

    do mc.SetSize nBiotac 23
    do mc.SetSize nVelocity 2

    let biotac =     mc.Var "Biotac"     [SizeSpec.symbol "BatchSize"; nBiotac]
    let optimalVel = mc.Var "OptimalVel" [SizeSpec.symbol "BatchSize"; nVelocity]
    let mlp = MLP.pars mc cfg.MLP

    let mi = mc.Instantiate DevCuda

    let pred = (MLP.pred mlp biotac.T).T
    let predFun = mi.Func pred |> arg biotac 

    let loss = MLP.loss mlp biotac.T optimalVel.T
    let lossFun = mi.Func loss |> arg2 biotac optimalVel

    let opt = GradientDescent.minimize {Step=cfg.StepSize} loss mi.ParameterVector
    let optFun = mi.Func opt |> arg2 biotac optimalVel   

    member this.Predict (biotac: Arrays) = predFun biotac

    member this.Train (dataset: Dataset<TactilePoint>) =
        printfn "Number of parameters in model: %d" (ArrayND.nElems mi.ParameterValues)

        // training minibatch generation
        let ds = TrnValTst.Of dataset
        printfn "%A" ds
        let trnBatches = ds.Trn.Batches cfg.BatchSize
        let tstBatch = ds.Tst.[0..10000-1]

        // save training set for testing
        //ds.Trn.ToHost().Save "TrnData.h5"
        //ds.Tst.ToHost().Save "TstData.h5"

        //let workDs = ds.Trn.[0..10000-1]

        // train
        mi.InitPars cfg.Seed

        let rng = System.Random()
        mi.ParameterStorage.Flat <- 
            ArrayNDHost.init mi.ParameterValues.Shape (fun () -> rng.NextDouble() * 0.2 - 0.1 |> single)
            |> ArrayNDCuda.toDev
        //ArrayND.fill (fun () -> rng.NextDouble() * 0.2 - 0.1 |> single)  mi.ParameterValues

        printfn "Training for %d iterations with batch size %d..." cfg.Iters cfg.BatchSize
        for itr = 0 to cfg.Iters do
            //let loss = lossFun workDs.Biotac workDs.OptimalVel
            //printfn "Training loss after %d iterations: %A" itr loss

            //optFun workDs.Biotac workDs.OptimalVel |> ignore  

            let loss = lossFun tstBatch.Biotac tstBatch.OptimalVel
            printfn "Test loss after %d iterations: %A" itr loss

            for trnBatch in trnBatches() do
                optFun trnBatch.Biotac trnBatch.OptimalVel |> ignore  
        printfn "Done."

    member this.Save filename = mi.SavePars filename     
    member this.Load filename = mi.LoadPars filename

    interface IController with
        member this.Predict biotac = this.Predict biotac
            

     
