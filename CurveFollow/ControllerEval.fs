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


let toArray = Movement.toArray

let curveAtPos (curve: XY list) (pos: float list) =
    let rec curveAtPos (curve: XY list) (pos: float list) cpos =
        match curve, pos with
        | _, [] -> List.rev cpos
        | [(_, cy)], _::rPos ->
            curveAtPos curve rPos (cy::cpos)
        | (cx1, cy1)::(cx2, _)::_, px::rPos when cx1 <= px && px < cx2 ->
            curveAtPos curve rPos (cy1::cpos)
        | (cx1, cy1)::_, px::rPos when px < cx1 ->
            curveAtPos curve rPos (cy1::cpos)
        | _::rCurve, _ ->
            curveAtPos rCurve pos cpos
        | _ -> failwith "empty curve"
    curveAtPos curve pos []


let plotControl (path: string) (curve: XY list) (recControl: ControlCurve.TactileControlCurve) =
    let curveX, curveY = toArray id curve
    let drivenPosX, drivenPosY = recControl.Points |> toArray (fun p -> p.Pos) 
    let predDistY = recControl.Points |> List.map (fun p -> p.EstDistance) |> Array.ofList
    let trgtY = recControl.Points |> List.map (fun p -> p.TargetY) |> Array.ofList
    let biotac = recControl.Points |> List.map (fun p -> p.Biotac)
    let curveDrivenY = curveAtPos curve (Array.toList drivenPosX) |> Array.ofList
    let distY = (curveDrivenY, drivenPosY) ||> Array.map2 (fun cx dx -> cx - dx)

    let left, right = Array.min drivenPosX, Array.max drivenPosX

    let dt = recControl.Points.[1].Time - recControl.Points.[0].Time
    let drivenVelY =
        drivenPosY
        |> Array.toSeq
        |> Seq.pairwise
        |> Seq.map (fun (a, b) -> (b - a) / dt)
        |> Seq.append (Seq.singleton 0.)
        |> Array.ofSeq

    R.pdf (path) |> ignore
    R.par2 ("oma", [0; 0; 0; 0])
    R.par2 ("mar", [3.2; 2.6; 1.0; 0.5])
    R.par2 ("mgp", [1.7; 0.7; 0.0])
    R.par2 ("mfrow", [4; 1])

    R.plot2 ([left; right], [curveY.[0] - 10.; curveY.[0] + 10.], "position", "x", "y")
    R.abline(h=curveY.[0]) |> ignore
    R.lines2 (curveX, curveY, "black")
    R.lines2 (drivenPosX, drivenPosY, "yellow")
    R.lines2 (drivenPosX, trgtY, "red")
    //R.lines2 (drivenPosX, curveDrivenY, "red")
    R.legend (125., curveY.[0] + 10., ["curve"; "driven"; "target"], col=["black"; "yellow"; "red"], lty=[1;1]) |> ignore

    R.plot2 ([left; right], [-15; 15], "velocity", "x", "y velocity")
    R.abline(h=0) |> ignore
    R.lines2 (drivenPosX, drivenVelY, "yellow")
    //R.legend (125., 15, ["control"; "driven"], col=["blue"; "yellow"], lty=[1;1]) |> ignore

    R.plot2 ([left; right], [-8; 8], "distance to curve", "x", "y distance")
    R.abline(h=0) |> ignore
    R.lines2 (drivenPosX, distY, "blue")
    R.lines2 (drivenPosX, predDistY, "red")
    R.legend (125., 8, ["true"; "predicted"], col=["blue"; "red"], lty=[1;1]) |> ignore

    // plot biotac
    let biotacImg = array2D biotac |> ArrayNDHost.ofArray2D |> ArrayND.transpose  // [chnl, smpl]
    let minVal, maxVal = ArrayND.minAxis 1 biotacImg, ArrayND.maxAxis 1 biotacImg
    let scaledImg = (biotacImg - minVal.[*, NewAxis]) / (maxVal - minVal).[*, NewAxis]
    R.image2 (ArrayNDHost.toArray2D scaledImg, lim=(0.0, 1.0),
              xlim=(left, right), colormap=Gray, title="biotac", xlabel="x", ylabel="channel")

    R.dev_off() |> ignore



/// Records data for all */movement.json files in the given directory.
let recordControl dir estDistanceFn =   
    for subDir in Directory.EnumerateDirectories dir do
        let distortionFile = Path.Combine (subDir, "distortion.dat")
        let recordFile = Path.Combine (subDir, "control.dat")
        if File.Exists distortionFile then

            printfn "%s" distortionFile
            let distortions : ControlCurve.ControlCurve = Pickle.load distortionFile
            let curve : XY list = Pickle.load (Path.Combine (subDir, "curve.dat"))

            let recControl = ControlCurve.record distortions estDistanceFn
            recControl |> Pickle.save recordFile 
            
            plotControl (Path.Combine (subDir, "control.pdf")) curve recControl


let plotRecordedControls dir =
    let bp = FsPickler.CreateBinarySerializer()
    
    for subDir in Directory.EnumerateDirectories dir do
        let recordedFile = Path.Combine (subDir, "control.dat")
        if File.Exists recordedFile then
            printfn "%s" recordedFile
            use tr = File.OpenRead recordedFile
            let recControl : ControlCurve.TactileControlCurve = bp.Deserialize tr
            use tr = File.OpenRead (Path.Combine (subDir, "curve.dat"))
            let curve : XY list = bp.Deserialize tr

            plotControl (Path.Combine (subDir, "control.pdf")) curve recControl
                        


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

