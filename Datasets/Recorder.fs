namespace Datasets

open System
open System.Diagnostics
open Microsoft.FSharp.Reflection

open Basics
open ArrayNDNS


[<AutoOpen>]
module RecorderTypes =

    /// Sensor interface.
    type ISensor =
        /// Data type of recorded data, e.g. single.
        abstract DataType : Type

    /// Sensor interface.
    type ISensor<'T> = 
        inherit ISensor
        /// Event that must be fired every time a sample was acquired.
        abstract SampleAcquired : IEvent<ArrayNDHostT<'T>>


    type internal IChannel =
        abstract FirstSampleTime: float
        abstract LastSampleTime: float
        abstract AverageSampleInterval: float
        abstract Clear: unit -> unit
        abstract SampleAt: float list -> IArrayNDHostT list
        abstract PrintStatistics: unit -> unit 

    type internal Channel<'T> (sensor:       ISensor<'T>,
                               startTime:    int64 ref,
                               recording:    bool ref) =

        let sampleTimes = ResizeArray<float>()
        let sampleDatas = ResizeArray<ArrayNDHostT<'T>>()

        let sampleAcquired (data: ArrayNDHostT<'T>) =
            if !recording then               
                let time = (float (Stopwatch.GetTimestamp() - !startTime)) / (float Stopwatch.Frequency)
                sampleTimes.Add time
                sampleDatas.Add data

        do sensor.SampleAcquired.Add sampleAcquired

        member this.Samples = sampleTimes |> List.ofSeq, sampleDatas |> List.ofSeq

        member this.Clear () = 
            sampleTimes.Clear()
            sampleDatas.Clear()

        member this.SampleAt times = 
            let rec interpolate (times: float list) (smplTimes: float list) 
                    (smplDatas: ArrayNDHostT<'T> list) (interpDatas: ArrayNDHostT<'T> list) = 
                match times, smplTimes, smplDatas with
                | [], _, _ ->
                    List.rev interpDatas
                | t::_, [], _ 
                | t::_, _, [] ->
                    failwithf "cannot extrapolate to time %f after last sample at time %f"
                        t (List.last smplTimes)
                | t::_, st::_, _ when t < st ->
                    failwithf "cannot extrapolate to time %f before first sample at time %f" t st
                | t::rTimes, st::stNext::_, data::dataNext::_ when st <= t && t <= stNext ->
                    let pos = (t - st) / (stNext - st) 
                    let oneMinusPos = 1. - pos
                    let pos = pos |> conv<'T> |> ArrayNDHost.scalar
                    let oneMinusPos = oneMinusPos |> conv<'T> |> ArrayNDHost.scalar
                    let ipData = oneMinusPos * data + pos * dataNext
                    interpolate rTimes smplTimes smplDatas (ipData::interpDatas)
                | _, _::rSmplTimes, _::rSmplDatas ->
                    interpolate times rSmplTimes rSmplDatas interpDatas

            interpolate times (sampleTimes |> List.ofSeq) (sampleDatas |> List.ofSeq) []

        member this.FirstSampleTime = sampleTimes.[0]
        member this.LastSampleTime = sampleTimes.[sampleTimes.Count - 1]
        member this.NSamples = sampleTimes.Count
        member this.AverageSampleInterval =
            (this.LastSampleTime - this.FirstSampleTime) / (float this.NSamples)

        member this.PrintStatistics () =
            printfn "Channel %A:  first sample time: %.3f s;  last sample time: %.3f s;  average interval: %.3f s"
                (sensor.GetType()) this.FirstSampleTime this.LastSampleTime this.AverageSampleInterval
            printfn "          number of samples: %d;  sampling rate: %.3f Hz"
                this.NSamples (1.0 / this.AverageSampleInterval)

        interface IChannel with
            member this.FirstSampleTime = this.FirstSampleTime
            member this.LastSampleTime = this.LastSampleTime
            member this.AverageSampleInterval = this.AverageSampleInterval
            member this.Clear () = this.Clear ()
            member this.SampleAt times = 
                this.SampleAt times
                |> List.map (fun x -> x :> IArrayNDHostT)
            member this.PrintStatistics () = this.PrintStatistics ()

        static member Create (sensor: ISensor) (startTime: int64 ref) (recording: bool ref) = 
            let t = typedefof<Channel<_>>.MakeGenericType(sensor.DataType)
            Activator.CreateInstance(t, sensor, startTime, recording) :?> IChannel


    /// Data recorder. Records samples of type 'S (must be a record type of ArrayNDHostTs) using
    /// the given sensors (must implemented the ISensor and ISensor<_> interfaces).
    type Recorder<'S> (sensors:     ISensor list) =

        // check the sensors match sample record type
        do 
            if not (FSharpType.IsRecord typeof<'S>) then
                failwith "Recorder sample type must be a record containing ArrayNDHostTs"
            let flds = FSharpType.GetRecordFields typeof<'S>
            if Seq.length flds <> Seq.length sensors + 1 then
                failwith "Sensor count does not match recorder sample type count"
            if not (flds.[0].Name = "Time" && flds.[0].PropertyType = typeof<ArrayNDHostT<single>>) then
                failwith "First field of recorder sample type must be \"Time: ArrayNDHostT<single>\""
            for fld, sensor in Seq.zip flds.[1..] sensors do
                let baseType = fld.PropertyType.GetGenericTypeDefinition()
                if baseType <> typedefof<ArrayNDHostT<_>> then
                    failwithf "All recorder sample type fields must be of type ArrayNDHostT<_> but \
                        field %s is of type %A" fld.Name fld.PropertyType
                let fldDataType = fld.PropertyType.GenericTypeArguments.[0]
                if fldDataType <> sensor.DataType then
                    failwithf "Field %s is of type %A but corresponding sensor records data of type %A"
                        fld.Name fld.PropertyType sensor.DataType

        /// start time of recording
        let startTime = ref (int64 0)

        /// currently recording?
        let recording = ref false

        /// recording channels
        let channels = 
            sensors
            |> List.map (fun s -> Channel<_>.Create s startTime recording)

        /// make a sample from given time and data arrays
        let makeSample (time: float) (data: IArrayNDHostT list) =
            let time = time |> single |> ArrayNDHost.scalar
            let values = (box time) :: (List.map box data) |> Array.ofList
            FSharpValue.MakeRecord (typeof<'S>, values) :?> 'S

        /// Starts recording.
        member this.Start () =
            if !recording then failwith "recording already started"
            startTime := Stopwatch.GetTimestamp ()            
            recording := true

        /// Stops recording.
        member this.Stop () =
            if not !recording then failwith "recording was not started"
            recording := false

        /// Clears all recorded data.
        member this.Clear () =
            if !recording then failwith "cannot clear while recording"
            for ch in channels do
                ch.Clear ()

        /// Gets all recorded samples using the specified inter-sample interval using linear
        /// interpolation.
        /// If None is specified for the interval, then the average sampling interval of the
        /// fastest sensor is used.
        member this.GetSamples (interval: float option) = 
            if !recording then failwith "cannot get samples while recording"

            // find start, stop time and interval
            let startTime =
                channels
                |> List.map (fun ch -> ch.FirstSampleTime)
                |> List.max
            let stopTime =
                channels
                |> List.map (fun ch -> ch.LastSampleTime)
                |> List.min
            let interval =
                match interval with
                | Some i -> i
                | None ->
                    // use inter-sample interval from fastest channel
                    channels
                    |> List.map (fun ch -> ch.AverageSampleInterval)
                    |> List.min

            // sample data at common times for all sensors
            let times = [ startTime .. interval .. stopTime ]
            let data = 
                channels 
                |> List.map (fun ch -> ch.SampleAt times |> Array.ofList)
                |> Array.ofList

            // convert data into samples
            seq {
                for idx, time in List.indexed times do
                    let chData = [for ch = 0 to List.length sensors - 1 do yield data.[ch].[idx]]
                    yield makeSample time chData
            } 

        /// Gets the recorded data as a Dataset<'S>. See GetSamples for details.
        member this.GetDataset (interval: float option) =
            this.GetSamples interval
            |> Dataset.FromSamples
        
        /// Prints recording statistics for all channels.
        member this.PrintStatistics () =
            printfn "Recorder for %A samples:" typeof<'S>
            for ch in channels do
                ch.PrintStatistics ()
