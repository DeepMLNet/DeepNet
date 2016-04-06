module RecorderTests

open Xunit
open FsUnit.Xunit

open System.Diagnostics
open System.Threading
open System.Timers

open Basics
open ArrayNDNS
open Datasets


type DummySensor<'T> (plusValue: int, interval: float) =
    let sw = Stopwatch.StartNew()
    let mutable cnt = 0

    let sampleEvent = new Event<'T>()

    let timer = new Timer (interval)
    do 
        timer.AutoReset <- true
        timer.Elapsed.Add (fun _ ->
            cnt <- cnt + 1
            cnt + plusValue
            |> conv<'T>
            |> sampleEvent.Trigger)
        timer.Start ()

    interface ISensor<'T> with
        member this.DataType = typeof<'T>
        member this.SampleAcquired = sampleEvent.Publish
        member this.Interpolate fac a b =
            let a = conv<float> a
            let b = conv<float> b
            let t = (1. - fac) * a + b
            conv<'T> t

    interface System.IDisposable with
        member this.Dispose () = timer.Dispose ()


type DummySample = {
    Time:       ArrayNDHostT<single>
    Sensor1:    ArrayNDHostT<single>
    Sensor2:    ArrayNDHostT<float>
}


[<Fact>]
let ``Test Recorder`` () =
    let sensors = [new DummySensor<single> (0, 30.0) :> ISensor; new DummySensor<float> (1000, 90.0) :> ISensor]
    let recorder = Recorder<DummySample> sensors
    recorder.Start ()
    Thread.Sleep 1000
    recorder.Stop ()
    recorder.PrintStatistics ()

    let smpls = recorder.GetSamples None
    printfn "Samples:"
    for smpl in smpls do
        printfn "Time: %f  Sensor1: %f  Sensor2: %f" 
            (ArrayND.value smpl.Time) (ArrayND.value smpl.Sensor1) (ArrayND.value smpl.Sensor2)

    
