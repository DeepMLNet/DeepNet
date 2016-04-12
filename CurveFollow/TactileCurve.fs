module TactileCurve


open System.Diagnostics
open System.Threading
open System.IO
open Nessos.FsPickler.Json

open Basics
open Datasets
open BRML.Drivers



/// Drive point.
type DrivePoint = {
    Time:               float
    Vel:                float * float
}

/// Curve to record.
type DriveCurve = {
    IndentorPos:        float
    StartPos:           float * float
    Accel:              float
    Points:             DrivePoint list
}

/// A tactile point.
type TactilePoint = {
    Time:               float
    Pos:                float * float
    Biotac:             int []
}

/// A tactile data curve.
type TactileCurve = {
    IndentorPos:        float
    Accel:              float
    Points:             TactilePoint list
}

/// Records a tactile curve given a drive curve.
let record (curve: DriveCurve) =

    let gotoStart withDown = 
        async {
            do! Devices.Linmot.DriveTo Devices.LinmotUpPos
            do! Devices.XYTable.DriveTo curve.StartPos
            if withDown then
                do! Devices.Linmot.DriveTo curve.IndentorPos
        } 
    let gotoStartTask = gotoStart true |> Async.StartAsTask
    
    Devices.XYTable.PosReportInterval <- 2
    let sensors = [Devices.XYTable :> ISensor; Devices.Biotac :> ISensor]
    let recorder = Recorder<TactilePoint> sensors

    let sw = Stopwatch()
    let rec control (points: DrivePoint list) =
        let t = (float sw.ElapsedMilliseconds) / 1000.
        match points with
        | _ when (let x, y = Devices.XYTable.CurrentPos in x > 142.) -> ()
        | [] -> ()
        | {Time=ct; Vel=vel}::({Time=ctNext}::_ as rPoints) when ct <= t && t < ctNext ->
            Devices.XYTable.DriveWithVel (vel, (curve.Accel, curve.Accel))  
            control rPoints
        | {Time=ct}::_ when t < ct ->
            //let dt = ct - t
            //if dt > 0.02 then Thread.Sleep (dt * 1000. |> int)
            control points
        | _::rCurve ->
            control rCurve

    gotoStartTask.Wait ()
    //exit 0

    recorder.Start ()
    sw.Start()
    control curve.Points
    recorder.Stop ()

    Devices.XYTable.Stop ()
    gotoStart false |> Async.Start

    {
        IndentorPos = curve.IndentorPos
        Accel       = curve.Accel
        Points      = recorder.GetSamples None |> List.ofSeq
    }
    



