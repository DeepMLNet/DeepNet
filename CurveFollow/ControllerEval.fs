module ControllerEval

open System
open System.IO
open RProvider
open RProvider.graphics
open RProvider.grDevices
open Nessos.FsPickler
open Nessos.FsPickler.Json

open Basics
open Basics.Cuda
open ArrayNDNS


type XY = float * float

type DistortionsAtXPosCfg = {
    MaxOffset:      float
    Hold:           float
    XPos:           float list
}

type Mode =
    | NoDistortions
    | DistortionsAtXPos of DistortionsAtXPosCfg


type Cfg = {
    Dx:             float
    VelX:           float
    Mode:           Mode
    FollowCurveToX: float
    IndentorPos:    float
}

type private DistortionTarget = {
    YPos:           float
    UntilXPos:      float
}

type private DistortionState = {
    XPos:           float list
    State:          DistortionTarget option  
}

let generate (cfg: Cfg) (rnd: System.Random) (curve: XY list) = 

    let baseX, baseY = curve.[0]
    let maxOffset = 9.6

    let rec generate curve x distState = seq {
        match curve with
        | [] -> ()
        | (x1,y1) :: (x2,y2) :: _ when x1 <= x && x < x2 ->
            // interpolate curve points
            let fac = (x2 - x) / (x2 - x1)
            let cy = fac * y1 + (1. - fac) * y2

            let ty, nextState =
                match cfg.Mode with
                | _ when x <= cfg.FollowCurveToX -> Some cy, distState
                | NoDistortions -> None, distState
                | DistortionsAtXPos dcfg ->
                    match distState.State with
                    | Some trgt ->
                        if x <= trgt.UntilXPos then Some trgt.YPos, distState
                        else None, {distState with State=None}
                    | None ->
                        match distState.XPos with
                        | dx::rXPos when dx <= x -> 
                            let trgt = cy + 2. * (rnd.NextDouble() - 0.5) * dcfg.MaxOffset
                            let trgt = min trgt (baseY + maxOffset)
                            let trgt = max trgt (baseY - maxOffset)
                            None, {XPos=rXPos; State=Some {YPos=trgt; UntilXPos=x + dcfg.Hold}}
                        | _ -> None, distState

            yield {ControlCurve.XPos=x; ControlCurve.YPos=ty}
            yield! generate curve (x + cfg.Dx) nextState
        | (x1, _) :: _ when x < x1 ->
            // x position is left of curve start
            yield! generate curve (x + cfg.Dx) distState
        | _ :: rCurve ->
            // move forward on curve
            yield! generate rCurve x distState
    }

    let initialState = 
        match cfg.Mode with
        | NoDistortions -> {XPos=[]; State=None}
        | DistortionsAtXPos dcfg -> {XPos=dcfg.XPos; State=None}
    let controlPnts = generate curve baseX initialState |> Seq.toList

    {
        ControlCurve.IndentorPos = cfg.IndentorPos
        ControlCurve.StartPos    = baseX, baseY
        ControlCurve.XVel        = cfg.VelX
        ControlCurve.Points      = controlPnts
    }


let generateDistortionsForFile cfgs path outDir =
    let rnd = Random ()
    let curves = Movement.loadCurves path

    for curveIdx, curve in List.indexed curves do      
        // generate controller distortion curves
        for cfgIdx, cfg in List.indexed cfgs do
            let dir = Path.Combine(outDir, sprintf "Curve%dCfg%d" curveIdx cfgIdx)
            Directory.CreateDirectory dir |> ignore

            let distortions = generate cfg rnd curve
            //plotControlCurve (Path.Combine (dir, "control.pdf")) curve cntrl

            if curveIdx <> 0 && curveIdx <> 6 then
                distortions |> Pickle.save (Path.Combine (dir, "distortion.dat"))
                curve |> Pickle.save (Path.Combine (dir, "curve.dat"))
            
type GenCfg = {
    CurveDir:           string
    DistortionDir:      string
    DistortionCfgs:     Cfg list
}

let generateDistortionsUsingCfg cfg  =
    for file in Directory.EnumerateFiles(cfg.CurveDir, "*.cur.npz") do
        printfn "%s" (Path.GetFullPath file)
        let outDir = Path.Combine(cfg.DistortionDir, Path.GetFileNameWithoutExtension file)
        generateDistortionsForFile cfg.DistortionCfgs file outDir


/// Records data for all */movement.json files in the given directory.
let recordControl dir estDistanceFn =   
    for subDir in Directory.EnumerateDirectories dir do
        let distortionFile = Path.Combine (subDir, "distortion.dat")
        let recordFile = Path.Combine (subDir, "recorded.dat")
        if File.Exists distortionFile then

            printfn "%s" distortionFile
            let distortions : ControlCurve.ControlCurve = Pickle.load distortionFile
            let curve : XY list = Pickle.load (Path.Combine (subDir, "curve.dat"))

            let recControl = ControlCurve.record distortions estDistanceFn
            //plotTactile (Path.Combine (subDir, "tactile.pdf")) curve tactileCurve
            recControl |> Pickle.save recordFile 

            //plotRecordedMovement (Path.Combine (subDir, "recorded.pdf")) curve recMovement None


let evalController (controllerCfg: Controller.Cfg) curveDir = 
    let mlpController = Controller.MLPController controllerCfg.MLPControllerCfg
    mlpController.Load controllerCfg.ModelFile
    
    let estDistanceFn biotac =
        CudaSup.setContext ()
        let biotacAry = biotac |> Array.map single |> ArrayNDHost.ofArray |> ArrayND.reshape [1; -1] |> ArrayNDCuda.toDev
        let predAry = mlpController.Predict biotacAry
        let pred = predAry.[[0; 0]] |> float
        pred

    recordControl curveDir estDistanceFn

