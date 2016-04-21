module ControlCurve

open System
open System.Diagnostics
open System.Threading

open Basics
open Datasets
open BRML.Drivers


type XY = float * float

/// Drive point.
type ControlPoint = {
    XPos:               float
    YPos:               float option
}

/// Curve to record.
type ControlCurve = {
    IndentorPos:        float
    StartPos:           XY
    XVel:               float
    Points:             ControlPoint list
}

/// A tactile point.
type TactileControlPoint = {
    Time:               float
    Pos:                XY
    Biotac:             float []
    EstDistance:        float
}

/// A tactile data curve.
type TactileControlCurve = {
    IndentorPos:        float
    Points:             TactileControlPoint list
}


type DistanceEstSensor () =
    let sampleAcquired = Event<float> ()
    interface ISensor<float> with
        member this.DataType = typeof<float>
        member this.SampleAcquired = sampleAcquired.Publish
        member this.Interpolate fac a b = (1.0 - fac) * a + fac * b

    member this.DistanceEstimated dist =
        sampleAcquired.Trigger dist

/// Records a tactile curve given a drive curve and a distance estimation function.
let record (curve: ControlCurve) (distanceEstFn: float [] -> float) =

    let gotoStart withDown = 
        async {
            do! Devices.Linmot.DriveTo Devices.LinmotUpPos
            do! Devices.XYTable.DriveTo curve.StartPos
            if withDown then
                do! Devices.Linmot.DriveTo curve.IndentorPos
        } 
    gotoStart true |> Async.RunSynchronously   

    let distEstSensor = DistanceEstSensor ()
    let mutable estDist = 0.0
    let estLock = obj ()
    use biotacHndlr = Devices.Biotac.SampleAcquired.Subscribe (fun biotac ->
        if Monitor.TryEnter estLock then
            estDist <- distanceEstFn biotac.Flat
            distEstSensor.DistanceEstimated estDist   
            Monitor.Exit estLock
    )

    Devices.XYTable.PosReportInterval <- 2
    let sensors = [Devices.XYTable :> ISensor
                   Devices.Biotac  :> ISensor
                   distEstSensor   :> ISensor]
    let recorder = Recorder<TactileControlPoint> sensors

    let sw = Stopwatch()
    let pidController = PID.Controller XYTableSim.pidCfg
    let rec control (points: ControlPoint list) =
        let t = (float sw.ElapsedMilliseconds) / 1000.
        let x, y = Devices.XYTable.CurrentPos 
        
        printf "t=%7.3f s     x=%7.3f mm     y=%.3f mm     EstDist=%7.3f mm     \r" t x y estDist
        if Console.KeyAvailable && Console.ReadKey().KeyChar = 'q' then
            Devices.XYTable.Stop();  exit 0

        match points with
        | _ when x > 142. -> ()
        | [] -> ()
        | {XPos=cx; YPos=tYPos}::{XPos=ncx; YPos=nYPos}::_ when cx <= x && x < ncx ->
            let yTrgt =
                match tYPos, nYPos with
                | Some tYPos, Some nYPos ->
                    // target position is overridden by control curve
                    let fac = (ncx - x) * (ncx - cx)
                    fac * tYPos + (1.-fac) * nYPos
                | _ ->
                    // distance is determined by model and we target zero distance
                    y - estDist
            let yVel = pidController.Control yTrgt y 
            Devices.XYTable.DriveWithVel ((curve.XVel, yVel))
            control points
        | {XPos=cx}::_ when x < cx ->
            // left of first point in curve
            Devices.XYTable.DriveWithVel ((curve.XVel, 0.0))
            control points
        | _::rCurve ->
            control rCurve

    recorder.Start ()
    sw.Start()
    control curve.Points
    recorder.Stop ()

    Devices.XYTable.Stop ()
    gotoStart false |> Async.Start
   
    printf "                                                                    \r"
    recorder.PrintStatistics ()

    {
        IndentorPos = curve.IndentorPos
        Points      = recorder.GetSamples None |> List.ofSeq
    }
    



