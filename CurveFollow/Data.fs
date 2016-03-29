module Data

open System
open System.IO
open Argu
open FSharp.Charting

open ArrayNDNS
open Datasets

type Arrays = ArrayNDT<single>
type Arrayd = ArrayNDT<double>
type Arrayb = ArrayNDT<bool>

type TactileCurve = {
    StartPos:           Arrays
    Indentor:           Arrays
    Time:               Arrays
    DrivenPos:          Arrays
    DrivenVel:          Arrays
    Biotac:             Arrays
    OptimalVel:         Arrays
    DistortionActive:   Arrayb
    DistortionPos:      Arrays
} with 
    member this.NSteps = (ArrayND.shape this.Time).[0]

type TactilePoint = {
    Time:               Arrays
    DrivenPos:          Arrays
    DrivenVel:          Arrays
    Biotac:             Arrays
    OptimalVel:         Arrays
    DistortionActive:   Arrayb
    DistortionPos:      Arrays   
}


let loadCurves srcDir =
    seq {
        for pageDir in Directory.EnumerateDirectories srcDir do
            for curveDir in Directory.EnumerateDirectories pageDir do
                yield async {
                    use curve   = NPZFile.Open (curveDir + "/curve.npz")
                    use tactile = NPZFile.Open (curveDir + "/tactile.npz")
                    return {
                        StartPos            = curve.Get "start_pos"             :> Arrayd |> ArrayND.mapTC single
                        Indentor            = curve.Get "indentor_pos"          :> Arrayd |> ArrayND.mapTC single
                        Time                = tactile.Get "time"                :> Arrayd |> ArrayND.mapTC single 
                        DrivenPos           = tactile.Get "pos"                 :> Arrayd |> ArrayND.mapTC single 
                        DrivenVel           = tactile.Get "vels"                :> Arrayd |> ArrayND.mapTC single
                        Biotac              = tactile.Get "biotac"              :> Arrayd |> ArrayND.mapTC single    
                        OptimalVel          = curve.Get "optimal_vels"          :> Arrayd |> ArrayND.mapTC single
                        DistortionActive    = curve.Get "distortion_actives"
                        DistortionPos       = curve.Get "distortion_poses"      :> Arrayd |> ArrayND.mapTC single
                    }
                }
    } |> Async.Parallel |> Async.RunSynchronously

let loadPoints srcDir = seq {
    for curve in loadCurves srcDir do 
        for step = 0 to curve.NSteps - 1 do
            yield {
                Time                = curve.Time.[step]
                DrivenPos           = curve.DrivenPos.[*, step]
                DrivenVel           = curve.DrivenVel.[*, step]
                Biotac              = curve.Biotac.[*, step]
                OptimalVel          = curve.OptimalVel.[*, step]
                DistortionActive    = curve.DistortionActive.[*, step]
                DistortionPos       = curve.DistortionPos.[*, step]           
            }           
} 

 
