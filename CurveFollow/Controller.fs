module Controller

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
    let pars = MLP.pars mc mlpHyperPars
    let md = mc.ParametersComplete ()
    let mi = md.Instantiate DevCuda

    let biotac = mc.Var "Biotac" [SizeSpec.fix 23; SizeSpec.fix batchSize]
    let pred = MLP.pred pars biotac
    let predFun = mi.Func (pred |> md.Subst) |> arg biotac

    interface IController with
        member this.Predict biotac = (predFun biotac.T).T

            

     
