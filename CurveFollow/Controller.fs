module Controller

open System

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
        let tstBatch = ds.Tst.All

        // train
        mi.InitPars cfg.Seed
        printfn "Training for %d iterations with batch size %d..." cfg.Iters cfg.BatchSize
        for itr = 0 to cfg.Iters do
            let loss = lossFun tstBatch.Biotac tstBatch.OptimalVel
            printfn "Test loss after %d iterations: %A" itr loss

            for trnBatch in trnBatches() do
                optFun trnBatch.Biotac trnBatch.OptimalVel |> ignore  
        printfn "Done."

    member this.Save filename = mi.SavePars filename     
    member this.Load filename = mi.LoadPars filename

    interface IController with
        member this.Predict biotac = this.Predict biotac
            

     
