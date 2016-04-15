namespace BRML.Drivers
//#nowarn "9"

open System
open System.Threading
//open System.Runtime.InteropServices

open TotalPhase


module Cheetah =

    [<Literal>]
    let MaxDevices = 16

    type Port = uint16
    type Uid = uint32
    type Handle = int
    type CheetahDev = CheetahDev of Port * Uid

    let findDevices () =
        let ports : Port [] = Array.zeroCreate MaxDevices
        let ids : Uid [] = Array.zeroCreate MaxDevices
        let nDevs = CheetahApi.ch_find_devices_ext (MaxDevices, ports, MaxDevices, ids)
        let ports = ports.[0 .. nDevs - 1] |> Array.toList
        let ids = ids.[0 .. nDevs - 1] |> Array.toList
        (ports, ids) ||> List.map2 (fun port id -> CheetahDev (port, id))

    let dumpDevices devs =
        printfn "Cheetah SPI devices:"
        if Seq.isEmpty devs then
            printfn "No cheetah devices found"
        else
            for CheetahDev(port, dev) in devs do
                printfn "Cheetah device at port %d with id %d" port dev

    let openDeviceById uid = 
        let devs = findDevices ()
        let dev = 
            devs 
            |> Seq.tryFind (fun (CheetahDev (_, devUid)) -> devUid = uid)
        match dev with
        | Some (CheetahDev (port, _)) ->
            let hnd : Handle = CheetahApi.ch_open (int port)
            if hnd <= 0 then failwithf "Opening cheetah device %d failed (already in use)" uid
            hnd
        | None ->
            dumpDevices devs
            failwithf "Cannot find cheetah device %d" uid

    let closeDevice (hnd: Handle) =
        CheetahApi.ch_close (hnd) |> ignore



[<AutoOpen>]
module Biotac =

    type BioTacChannelT = int
    type BioTacValueT = float

    type BioTacSampleT = {
        Flat:           BioTacValueT []
        ByChannel:      Map<BioTacChannelT, BioTacValueT>
    }

    type BioTacCfgT = {
        Cheetah:        Cheetah.Uid
        Index:          int
    }

    /// Biotac sensor
    type BiotacT (config: BioTacCfgT) =

        let SpiBitrate =        4400          // kHz
        let AfterSampleDelay =  50000         // ns
        let AfterReadDelay =    20000         // ns
        let MaxValue =          8192          // maximum data value

        /// sensor query frame
        let QueryFrame = 
            [0x80; 0x83; 0x85; 0x86; 0xa2; 0xa4; 0xa7; 0xa8; 0xab; 0xad; 0xae; 0xb0;
             0xb3; 0xb5; 0xb6; 0xb9; 0xba; 0xbc; 0xbf; 0xc1; 0xc2; 0xc4; 0xc7 ]

        /// chip select on for specified biotac
        let ssOn =
            match config.Index with
            | 0 -> byte 1
            | 1 -> byte 2
            | 2 -> byte 4
            | _ -> failwith "invalid biotac index"

        /// chip select off
        let ssOff = byte 0

        [<VolatileField>]
        let mutable currentSample = None

        let mutable serial = None

        let sampleAcquired = new Event<BioTacSampleT>()

        let mutable acquisitionEnabled = true

        let acquisitionThread = Thread(fun () ->
            let cheetah = Cheetah.openDeviceById config.Cheetah

            // configure SPI settings
            CheetahApi.ch_spi_configure (cheetah, 
                                         CheetahSpiPolarity.CH_SPI_POL_RISING_FALLING,
                                         CheetahSpiPhase.CH_SPI_PHASE_SAMPLE_SETUP,
                                         CheetahSpiBitorder.CH_SPI_BITORDER_MSB,
                                         byte 0) |> ignore
            CheetahApi.ch_spi_bitrate (cheetah, SpiBitrate) |> ignore
            CheetahApi.ch_spi_queue_oe(cheetah, byte 1) |> ignore   // not a queue function!

            // reset biotac
            CheetahApi.ch_target_power (cheetah, CheetahApi.CH_TARGET_POWER_OFF) |> ignore
            Thread.Sleep(1000)
            CheetahApi.ch_target_power (cheetah, CheetahApi.CH_TARGET_POWER_ON) |> ignore
            Thread.Sleep(1000)            

            // query serial number
            CheetahApi.ch_spi_queue_clear (cheetah) |> ignore
            CheetahApi.ch_spi_queue_ss (cheetah, ssOn) |> ignore
            CheetahApi.ch_spi_queue_byte (cheetah, 1, byte 0x61) |> ignore  // read command
            CheetahApi.ch_spi_queue_byte (cheetah, 1, byte 0x15) |> ignore  // serial number
            CheetahApi.ch_spi_queue_ss (cheetah, ssOff) |> ignore
            CheetahApi.ch_spi_queue_delay_ns (cheetah, AfterSampleDelay) |> ignore
            CheetahApi.ch_spi_batch_shift (cheetah, 0, [||]) |> ignore

            // read response
            CheetahApi.ch_spi_queue_clear (cheetah) |> ignore
            CheetahApi.ch_spi_queue_ss (cheetah, ssOn) |> ignore
            CheetahApi.ch_spi_queue_byte (cheetah, 1, byte 0) |> ignore 
            let serialBuf = 
                seq {
                    for i = 1 to 100 do
                        let inBuf = [| byte 0 |]
                        CheetahApi.ch_spi_batch_shift (cheetah, 1, inBuf) |> ignore
                        yield inBuf.[0]
                } 
                |> Seq.cache 
                |> Seq.takeWhile (fun b -> b <> byte 0x00 && b <> byte 0xff)
                |> Seq.map char
                |> Seq.toArray
                |> fun x -> String(x)
            CheetahApi.ch_spi_queue_clear (cheetah) |> ignore
            CheetahApi.ch_spi_queue_ss (cheetah, ssOff) |> ignore
            CheetahApi.ch_spi_batch_shift (cheetah, 0, [||]) |> ignore

            // check if biotac is present
            if not (serialBuf.Length > 2 && serialBuf.[0..1] = "BT") then
                failwithf "biotac not detected on cheetah %d at index %d" config.Cheetah config.Index
            serial <- Some serialBuf

            // build cheetah SPI batch for obtaining a sample from the biotac
            CheetahApi.ch_spi_queue_clear (cheetah) |> ignore
            for cmd in QueryFrame do
                // send sampling command
                CheetahApi.ch_spi_queue_ss (cheetah, ssOn) |> ignore
                CheetahApi.ch_spi_queue_byte (cheetah, 1, byte cmd) |> ignore 
                CheetahApi.ch_spi_queue_byte (cheetah, 1, byte 0x00) |> ignore 
                CheetahApi.ch_spi_queue_ss (cheetah, ssOff) |> ignore
                CheetahApi.ch_spi_queue_delay_ns (cheetah, AfterSampleDelay) |> ignore

                // read response
                CheetahApi.ch_spi_queue_ss (cheetah, ssOn) |> ignore
                CheetahApi.ch_spi_queue_byte (cheetah, 2, byte 0x00) |> ignore 
                CheetahApi.ch_spi_queue_ss (cheetah, ssOff) |> ignore
                CheetahApi.ch_spi_queue_delay_ns (cheetah, AfterReadDelay) |> ignore
            let responseLength = CheetahApi.ch_spi_batch_length (cheetah)

            // queue initial batches
            for i = 0 to 4 do 
                CheetahApi.ch_spi_async_submit (cheetah) |> ignore

            // acquisition loop
            while acquisitionEnabled do
                // get batch result
                let response : byte [] = Array.zeroCreate responseLength
                CheetahApi.ch_spi_async_collect (cheetah, responseLength, response) |> ignore

                // parse data
                let mutable sampleValid = true
                let dataFrame =
                    seq {
                        for pos in {2 .. 4 .. responseLength - 1} do
                            let msb, lsb = response.[pos], response.[pos+1]

                            // check parity
                            let parity (x: byte) =
                                seq {
                                    let mutable x = x
                                    for b = 0 to 7 do
                                        yield x &&& 1uy
                                        x <- x >>> 1 
                                    }
                                |> Seq.fold (fun s b -> s ^^^ b) 0uy
                            if parity msb <> 1uy || parity lsb <> 1uy then
                                //printfn "received invalid biotac sample: pos=%d data=0x%x%x" pos msb lsb
                                sampleValid <- false

                            let msb, lsb = uint16 msb, uint16 lsb
                            let highData, lowData = msb >>> 1, lsb >>> 3
                            yield float ((highData <<< 5) ||| lowData) / float MaxValue
                    }
                    |> Array.ofSeq
                assert (Array.length dataFrame = List.length QueryFrame)
                let sample = {
                    Flat      = dataFrame
                    ByChannel = Seq.zip QueryFrame dataFrame |> Map.ofSeq
                }

                // save data and fire event
                if sampleValid then
                    currentSample <- Some sample
                    async { sampleAcquired.Trigger sample } |> Async.Start

                // queue next batch
                CheetahApi.ch_spi_async_submit (cheetah) |> ignore

            // shutdown biotac and cheetah
            CheetahApi.ch_spi_queue_oe(cheetah, byte 0) |> ignore   // not a queue function!
            CheetahApi.ch_target_power (cheetah, CheetahApi.CH_TARGET_POWER_OFF) |> ignore
            Cheetah.closeDevice cheetah
        )
            
        do acquisitionThread.IsBackground <- true
        do acquisitionThread.Start()
            
        /// a new sample has been acquired
        member this.SampleAcquired = sampleAcquired.Publish

        /// the current sample
        member this.CurrentSample =
            if not acquisitionEnabled then failwith "Biotac has been disposed"
            match currentSample with
            | Some smpl -> smpl
            | None -> Async.AwaitEvent (this.SampleAcquired) |> Async.RunSynchronously

        /// the next fresh sample
        member this.GetNextSample () =
            Async.AwaitEvent (this.SampleAcquired) |> Async.RunSynchronously

        /// Biotac serial number
        member this.Serial = serial

        interface IDisposable with
            member this.Dispose() =               
                acquisitionEnabled <- false
                acquisitionThread.Join()

        interface Datasets.RecorderTypes.ISensor<BioTacValueT []> with
            member this.DataType = typeof<BioTacValueT []>
            member this.SampleAcquired =
                this.SampleAcquired
                |> Event.map (fun smpl -> smpl.Flat)
            member this.Interpolate fac a b =
                (a, b)
                ||> Array.map2 (fun a b -> (1.0 - fac) * a + fac * b)


