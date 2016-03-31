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


type MLPController (mlpHyperPars:    MLP.HyperPars,
                    batchSize:       int) =

    let mc = ModelBuilder<single> "MLPController"

    let nBiotac = SizeSpec.symbol "nBiotac"
    let nVelocity = SizeSpec.symbol "nVelocity"
    do mc.SetSize nBiotac 23
    do mc.SetSize nVelocity 2

    let biotac =     mc.Var "Biotac"     [SizeSpec.fix batchSize; nBiotac]
    let optimalVel = mc.Var "OptimalVel" [SizeSpec.fix batchSize; nVelocity]
    let mlp = MLP.pars mc mlpHyperPars

    let mi = mc.Instantiate DevCuda

    let pred = MLP.pred mlp biotac.T
    let predFun = mi.Func pred |> arg biotac 

    let loss = MLP.loss mlp biotac.T optimalVel.T
    let lossFun = mi.Func loss |> arg2 biotac optimalVel

    let opt = GradientDescent.minimize {Step=1e-5f} loss mi.ParameterVector
    let optFun = mi.Func opt |> arg2 biotac optimalVel   

    interface IController with
        member this.Predict biotac = (predFun biotac.T).T

    member this.Train (dataset: Dataset<TactilePoint>) =
        printfn "Number of parameters in model: %d" (ArrayND.nElems mi.ParameterValues)

        // TODO: 
        //       - relax requirement that all sizes must be known, so that only
        //         ParameterSet relevant sizes must be known and batch size can change, i.e. dynamic compilation

        // initialize parameters
        let rng = Random (10)
        let initPars : ArrayNDHostT<single> = 
            ArrayNDHost.zeros mi.ParameterValues.Shape
            |> ArrayND.map (fun _ -> rng.NextDouble() - 0.5 |> single)    
        mi.ParameterValues.[Fill] <- initPars |> ArrayNDCuda.toDev   

        // training minibatch generation
        let ds = TrnValTst.Of dataset
        let trnBatches = ds.Trn.Batches 10000
        let tmpl = trnBatches () |> Seq.head
        let tstBatch = trnBatches() |> Seq.head

        let iters = 100
        printfn "Training for %d iterations..." iters
        for itr = 0 to iters do
            let loss = lossFun tstBatch.Biotac tstBatch.OptimalVel
            printfn "Loss after %d iterations: %A" itr loss

            for trnBatch in trnBatches() do
                optFun trnBatch.Biotac trnBatch.OptimalVel |> ignore  

    member this.Save filename = mi.SavePars filename     
    member this.Load filename = mi.LoadPars filename
            

     
