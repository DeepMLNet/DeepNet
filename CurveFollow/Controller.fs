module Controller

open System
open System.IO
open Nessos.FsPickler

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
    let lossFun = mi.Func loss |> arg2 biotac target

    let optimizer = GradientDescent DevCuda
    let opt = optimizer.Minimize loss mi.ParameterVector
    let optFun = mi.Func (opt, loss) |> optimizer.Cfg |> arg2 biotac target   

    member this.Predict (biotac: Arrays) = 
        predFun biotac

    member this.Train (dataset: TrnValTst<FollowSample>) (cfg: Train.Cfg) =
        let lossFn fs = lossFun fs.Biotac fs.YDist |> ArrayND.value
        let optFn lr fs = 
            let _, loss = optFun fs.Biotac fs.YDist {GradientDescent.Step=lr}
            lazy (ArrayND.value loss)
        Train.train mi lossFn optFn dataset cfg


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
}

    
let loadRecordedMovementAsDataset baseDirs downsampleFactor = 
    seq {
        let s = FsPickler.CreateBinarySerializer()
        for baseDir in baseDirs do
            for dir in Directory.EnumerateDirectories baseDir do
                let recordedFile = Path.Combine (dir, "recorded.dat")
                if File.Exists recordedFile then
                    use f = File.OpenRead recordedFile
                    let recorded : Movement.RecordedMovement = s.Deserialize f           
                    for rmp in recorded.Points do
                        yield {
                            FollowSample.Biotac = rmp.Biotac |> Array.map single |> ArrayNDHost.ofArray
                            FollowSample.YDist  = rmp.YDist |> single 
                                                  |> ArrayNDHost.scalar |> ArrayND.reshape [1]
                        }
    }            
    |> Seq.everyNth downsampleFactor
    |> Dataset.FromSamples
     

let loadDataset (cfg: Cfg) : TrnValTst<FollowSample> =
    match cfg.DatasetCache with
    | Some filename when File.Exists (filename + "-Trn.h5")
                      && File.Exists (filename + "-Val.h5")
                      && File.Exists (filename + "-Tst.h5") ->
        printfn "Using cached dataset %s" (Path.GetFullPath filename)
        TrnValTst.Load filename
    | _ ->
        let dataset = {
            TrnValTst.Trn = loadRecordedMovementAsDataset cfg.TrnDirs cfg.DownsampleFactor
            TrnValTst.Val = loadRecordedMovementAsDataset cfg.ValDirs cfg.DownsampleFactor
            TrnValTst.Tst = loadRecordedMovementAsDataset cfg.TstDirs cfg.DownsampleFactor
        }
        match cfg.DatasetCache with
        | Some filename -> dataset.Save filename
        | _ -> ()
        dataset

        
     
let train (cfg: Cfg) =
    let dataset = loadDataset cfg |> TrnValTst.ToCuda
    let mlpController = MLPController cfg.MLPControllerCfg
    mlpController.Train dataset cfg.TrainCfg |> ignore
    mlpController.Save "model.h5"


let saveChart (path: string) (chart:FSharp.Charting.ChartTypes.GenericChart) =
    use control = new FSharp.Charting.ChartTypes.ChartControl(chart)
    control.Size <- Drawing.Size(1280, 720)
    chart.CopyAsBitmap().Save(path, Drawing.Imaging.ImageFormat.Png)

let plot (cfg: Cfg) =
//    let mlpController = Controller.MLPController cfg.MLPControllerCfg
//    mlpController.Load "model.h5"
//
//    let dataset = loadCurveDataset cfg.DatasetDir   
//    for idx, smpl in Seq.indexed (dataset |> Seq.take 10) do
//        let predVel = mlpController.Predict smpl.Biotac
//
//        let posChart = 
//            Chart.Line (Seq.zip smpl.Time smpl.DrivenPos.[*, 1])
//            |> Chart.WithXAxis (Title="Time")
//            |> Chart.WithYAxis (Title="Position", Min=0.0, Max=12.0)
//        
//        let controlCharts =
//            Chart.Combine(
//                [Chart.Line (Seq.zip smpl.Time predVel.[*, 1], Name="pred")
//                 Chart.Line (Seq.zip smpl.Time smpl.OptimalVel.[*, 1], Name="optimal")
//                 Chart.Line (Seq.zip smpl.Time smpl.DrivenVel.[*, 1], Name="driven") ])
//            |> Chart.WithXAxis (Title="Time")
//            |> Chart.WithYAxis (Title="Velocity", Min=(-2.0), Max=2.0)
//            |> Chart.WithLegend ()
//
//        let chart = Chart.Rows [posChart; controlCharts]
//        chart |> saveChart (sprintf "curve%03d.png" idx)   
    ()

