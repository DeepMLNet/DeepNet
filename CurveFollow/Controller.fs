module Controller

open System
open System.IO
open Nessos.FsPickler
open Nessos.FsPickler.Json

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


type FollowSample = {
    Biotac:          Arrays
    YDist:           Arrays
}


type MLPControllerCfg = {
    MLP:            MLP.HyperPars
}

let nBiotac = SizeSpec.symbol "nBiotac"
let nTarget = SizeSpec.symbol "nTarget"


    
type MLPController (cfg:   MLPControllerCfg) =

    let mc = ModelBuilder<single> "MLPController"

    do mc.SetSize nBiotac 23
    do mc.SetSize nTarget 1

    let biotac =  mc.Var "Biotac"  [SizeSpec.symbol "BatchSize"; nBiotac]
    let target =  mc.Var "Target"  [SizeSpec.symbol "BatchSize"; nTarget]
    let mlp = MLP.pars mc cfg.MLP

    let mi = mc.Instantiate DevCuda
    do printfn "Number of parameters in MLPController: %d" (ArrayND.nElems mi.ParameterValues)

    let pred = (MLP.pred mlp biotac.T).T
    let predFun = mi.Func pred |> arg biotac 

    let loss = MLP.loss mlp biotac.T target.T

    member this.Predict (biotac: Arrays) = 
        predFun biotac

    member this.Train (dataset: TrnValTst<FollowSample>) (trainCfg: Train.Cfg) =
        let opt = Adam (loss, mi.ParameterVector, DevCuda)
        let optCfg = opt.DefaultCfg

        let trainable = 
            Train.trainableFromLossExpr mi loss 
                (fun fs -> VarEnv.ofSeq [biotac, fs.Biotac; target, fs.YDist]) opt optCfg
        Train.train trainable dataset trainCfg

    member this.Save filename = mi.SavePars filename     
    member this.Load filename = mi.LoadPars filename

    interface IController with
        member this.Predict biotac = this.Predict biotac
            

// configuration
type Cfg = {
    TrnDirs:            string list
    ValDirs:            string list
    TstDirs:            string list
    DatasetCache:       string option
    DownsampleFactor:   int
    MLPControllerCfg:   MLPControllerCfg   
    TrainCfg:           Train.Cfg 
    ModelFile:          string
    TrnResultFile:      string
}


let loadRecordedMovements baseDirs =
    let s = FsPickler.CreateBinarySerializer()
    seq {
        for baseDir in baseDirs do
            for dir in Directory.EnumerateDirectories baseDir do
                let recordedFile = Path.Combine (dir, "recorded.dat")
                if File.Exists recordedFile then
                    use f = File.OpenRead recordedFile
                    let recorded : Movement.RecordedMovement = s.Deserialize f           
                    yield recorded
    }
    
let recordedMovementAsFollowSamples (recorded: Movement.RecordedMovement) = seq {
    for rmp in recorded.Points do
        yield {
            FollowSample.Biotac = rmp.Biotac |> Array.map single |> ArrayNDHost.ofArray
            FollowSample.YDist  = rmp.YDist |> single 
                                  |> ArrayNDHost.scalar |> ArrayND.reshape [1]
        }    
}

let loadRecordedMovementAsCurveDataset baseDirs =
    seq {
        for recorded in loadRecordedMovements baseDirs do
            let nSteps = List.length recorded.Points
            let nChannels = Array.length recorded.Points.[0].Biotac
                    
            let biotac = ArrayNDHost.zeros [nSteps; nChannels]
            let ydist = ArrayNDHost.zeros [1; nChannels]

            for step, rmp in List.indexed recorded.Points do
                biotac.[step, *] <- rmp.Biotac |> Array.map single |> ArrayNDHost.ofArray
                ydist.[[step; 0]] <- rmp.YDist |> single

            yield {FollowSample.Biotac=biotac; FollowSample.YDist=ydist}                    
    }            
    |> Dataset.FromSamples


let loadRecordedMovementAsPointDataset baseDirs downsampleFactor = 
    seq {
        for recorded in loadRecordedMovements baseDirs do    
            yield! recordedMovementAsFollowSamples recorded     
    }            
    |> Seq.everyNth downsampleFactor
    |> Dataset.FromSamples
     

let loadPointDataset (cfg: Cfg) : TrnValTst<FollowSample> =
    match cfg.DatasetCache with
    | Some filename when File.Exists (filename + "-Trn.h5")
                      && File.Exists (filename + "-Val.h5")
                      && File.Exists (filename + "-Tst.h5") ->
        printfn "Using cached dataset %s" (Path.GetFullPath filename)
        TrnValTst.Load filename
    | _ ->
        let dataset = {
            TrnValTst.Trn = loadRecordedMovementAsPointDataset cfg.TrnDirs cfg.DownsampleFactor
            TrnValTst.Val = loadRecordedMovementAsPointDataset cfg.ValDirs cfg.DownsampleFactor
            TrnValTst.Tst = loadRecordedMovementAsPointDataset cfg.TstDirs cfg.DownsampleFactor
        }
        match cfg.DatasetCache with
        | Some filename -> dataset.Save filename
        | _ -> ()
        dataset

     
let train (cfg: Cfg) =
    printfn "Using configuration:\n%A" cfg
    let dataset = loadPointDataset cfg |> TrnValTst.ToCuda
    let mlpController = MLPController cfg.MLPControllerCfg
    let trnRes = mlpController.Train dataset cfg.TrainCfg     
    trnRes.Save cfg.TrnResultFile
    mlpController.Save cfg.ModelFile

let saveChart (path: string) (chart:FSharp.Charting.ChartTypes.GenericChart) =
    use control = new FSharp.Charting.ChartTypes.ChartControl(chart)
    control.Size <- Drawing.Size(1280, 720)
    chart.CopyAsBitmap().Save(path, Drawing.Imaging.ImageFormat.Png)



let plotCurvePredictions (cfg: Cfg) curveDir =
    let bp = FsPickler.CreateBinarySerializer()
    let jp = FsPickler.CreateJsonSerializer(indent=true)

    let mlpController = MLPController cfg.MLPControllerCfg
    mlpController.Load cfg.ModelFile
    
    for subDir in Directory.EnumerateDirectories curveDir do
        let recordedFile = Path.Combine (subDir, "recorded.dat")
        if File.Exists recordedFile then 
            printfn "%s" recordedFile
            use tr = File.OpenRead recordedFile
            let recMovement : Movement.RecordedMovement = bp.Deserialize tr
            use tr = File.OpenRead (Path.Combine (subDir, "curve.dat"))
            let curve : Movement.XY list = bp.Deserialize tr

            // predict
            let ds = recordedMovementAsFollowSamples recMovement |> Dataset.FromSamples 
            let biotac = ds.All.Biotac :?> ArrayNDHostT<_> |> ArrayNDCuda.toDev
            let pred = mlpController.Predict biotac :?> ArrayNDCudaT<_>
            let predDistY = pred.[*, 0] |> ArrayNDCuda.toHost |> ArrayNDHost.toList |> List.map float

            // save and plot
            use tw = File.CreateText (Path.Combine (subDir, "predicted.json"))
            jp.Serialize (tw, predDistY)
            Movement.plotRecordedMovement (Path.Combine (subDir, "predicted.pdf")) curve recMovement (Some predDistY)


