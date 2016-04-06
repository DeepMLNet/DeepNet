module Record


open System.Diagnostics
open System.Threading

open Basics
open ArrayNDNS
open Datasets
open BRML.Drivers

type Arrays = ArrayNDT<single>


type DriveCurve = {
    Indentor:           Arrays
    StartPos:           Arrays
    Accel:              Arrays
    Time:               Arrays
    Vel:                Arrays
}

type private DrivePoint = {
    Time:               int64
    Vel:                XYTable.XYTupleT
}


type DrivenCurve = {
    Time:               Arrays
    Pos:                Arrays
    Biotac:             Arrays
}


let recordCurve (curve: DriveCurve) =
    let indentorPos = curve.Indentor.[[0]] |> float
    let accel = curve.Accel.[[0]] |> float, curve.Accel.[[1]] |> float
    let startPos = curve.StartPos.[[0]] |> float, curve.StartPos.[[1]] |> float

    let gotoStart withDown = 
        async {
            do! Devices.Linmot.DriveTo (Devices.LinmotUpPos)
            do! Devices.XYTable.DriveTo (startPos)
            if withDown then
                do! Devices.Linmot.DriveTo (indentorPos)
        } 
    let gotoStartTask = gotoStart true |> Async.StartAsTask

    let sensors = [Devices.XYTable :> ISensor; Devices.Biotac :> ISensor]
    let recorder = Recorder<DrivenCurve> sensors

    let curve =
        seq {
            for p = 0 to curve.Time.Shape.[0] - 1 do
                yield {
                    DrivePoint.Time = curve.Time.[[p]] * 1000.f |> int64
                    DrivePoint.Vel  = float curve.Vel.[[p; 0]], float curve.Vel.[[p; 1]] 
                }   
        }
        |> List.ofSeq

    let sw = Stopwatch()
    let rec control (curve: DrivePoint list) =
        let t = sw.ElapsedMilliseconds
        match curve with
        | [] -> ()
        | {Time=ct}::_ when t < ct ->
            let dt = ct - t
            if dt > 20L then Thread.Sleep (int dt)
            control curve
        | {Time=ct; Vel=vel}::({Time=ctNext}::_ as rCurve) when ct <= t && t < ctNext ->
            Devices.XYTable.DriveWithVel (vel, accel)  
            control rCurve
        | _::rCurve ->
            control rCurve

    gotoStartTask.Wait ()

    recorder.Start ()
    control curve
    recorder.Stop ()

    Devices.XYTable.Stop ()
    gotoStart false |> Async.Start

    recorder.GetDataset(None).All



