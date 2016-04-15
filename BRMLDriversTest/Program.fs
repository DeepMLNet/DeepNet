open BRML.Drivers

open System
open System.Threading
open System.Diagnostics

let demoTable (xyTable: XYTableT) = async {
    //printf "Homing XYTable..."
    do! xyTable.Home() 
    //printfn "Done."

    //printfn "Drive to center"
    do! xyTable.DriveTo((70., 70.)) 

    //printfn "Drive with constant velocity"
    xyTable.DriveWithVel ((10., 10.)) 

    do! Async.Sleep 2000

    //printfn "Stop."
    xyTable.Stop ()

    let rng = Random()

    for i = 1 to 20 do
        let xpos = rng.NextDouble() * 147.
        let ypos = rng.NextDouble() * 140.
        let pos = xpos, ypos
        //printfn "Drive to %A" pos
        do! xyTable.DriveTo(pos)

    //printfn "Exception..."
    //failwith "my error"
}

let demoTablePos (xyTable: XYTableT) = 
    xyTable.Home() |> Async.RunSynchronously
    xyTable.DriveTo((70., 70.)) |> Async.RunSynchronously

    //printfn "Drive with constant velocity"
    xyTable.DriveWithVel ((3., -5.)) 

    let rng = Random()
    let rndSpeed () = (rng.NextDouble() - 0.5) * 10.0
    //let rndSpeed () = rng.NextDouble() * 10.0

    let nCntrlSamples = 300
    let nPosSamples = ref 0
    let sw = System.Diagnostics.Stopwatch.StartNew()

    let showPos = ref true

    //xyTable.PosSampleInterval <- 60
    xyTable.PosAcquired |> Event.add (fun (xpos, ypos) ->
        if !showPos then
            //printf "Position: x=%f    y=%f    \r" xpos ypos
            nPosSamples := !nPosSamples + 1
    )

    let ticksPerUs = Stopwatch.Frequency / (1000L * 1000L)

    let mutable xvel, yvel = 0.0, 0.0

    for i = 1 to nCntrlSamples do
        //let xpos, ypos = xyTable.GetNextPos() 
        //printf "Position: x=%f    y=%f    \r" xpos ypos

        xvel <- xvel + 0.1 * rndSpeed()
        yvel <- yvel + 0.1 * rndSpeed()
        //Thread.Sleep(10)
        xyTable.DriveWithVel ((xvel, yvel))


        
    showPos := false
    printfn ""
    let duration = sw.ElapsedMilliseconds
    let cntrlFreq = (float nCntrlSamples) / ((float duration) / 1000.)
    let posFreq = (float !nPosSamples) / ((float duration) / 1000.)
    printfn "XYTable control  rate: %f Hz" cntrlFreq
    printfn "XYTable position rate: %f Hz" posFreq
    printfn ""

    xyTable.Stop()
    
    //let sendDur = float Stepper.sendTime / float Stepper.sendReqs
    //let recvDur = float Stepper.recvTime / float Stepper.recvReqs
    //printfn "Send time per req: %f ms" sendDur
    //printfn "Recv time per req: %f ms" recvDur


let demoLinmot (linmot: LinmotT) = async {
    let rng = Random()
    for i = 1 to 20 do
        let pos = rng.NextDouble() * (-20.)
        //printfn "Drive to %f" pos
        do! linmot.DriveTo (pos)

    //printfn "Power off"
    //do! linmot.Power false 
}

let demoBiotac (biotac: BiotacT) =
    let nSamples = 1000

    printf "Waiting for biotac initialization..."
    biotac.GetNextSample () |> ignore
    printfn "Done."
    printfn "Biotac serial: %s" biotac.Serial.Value

    let sw = System.Diagnostics.Stopwatch.StartNew()
    for i = 1 to nSamples do
        biotac.GetNextSample() |> ignore
    let duration = sw.ElapsedMilliseconds
    let freq = (float nSamples) / ((float duration) / 1000.)
    printfn "Biotac sampling rate: %f Hz" freq
    printfn ""

    (*
    let sx, sy = Console.CursorLeft, Console.CursorTop
    while not Console.KeyAvailable do
        Console.CursorLeft <- sx
        Console.CursorTop <- sy

        let smpl = biotac.GetNextSample()
        for chnl in smpl.Flat do
            printf "%04x " chnl  
        printfn ""
    *)


let testTbl () =
    Devices.XYTable.Home () |> Async.RunSynchronously

    Devices.XYTable.DriveTo ((70., 70.)) |> Async.RunSynchronously
    Devices.XYTable.DriveWithVel ((1.0, -2.0))

    let mutable rPoses = 0
    Devices.XYTable.PosAcquired.Add (fun pos ->
        printfn "pos: %A" pos        
        rPoses <- rPoses + 1
    )

    let sw = Stopwatch.StartNew()
    while sw.ElapsedMilliseconds < 10000L do
        Thread.Sleep(100)
        //printfn "pos: %A" Devices.XYTable.CurrentPos

    let freq = float rPoses / (float sw.ElapsedMilliseconds / 1000.)
    printfn "report freq: %f Hz" freq

    Devices.XYTable.Stop ()


[<EntryPoint>]
let main argv = 
    use linmot = Devices.Linmot
    use xyTable = Devices.XYTable
    use biotac = Devices.Biotac

    printf "Homing Linmot..."
    linmot.Home(false) |> Async.RunSynchronously
    linmot.DriveTo(-1.) |> Async.RunSynchronously
    printfn "Done."    

    //let demoLinmot = demoLinmot linmot |> Async.StartAsTask  
    //let demoTbl = async { demoTablePos xyTable } |> Async.StartAsTask
    let demoBt = async { demoBiotac biotac } |> Async.StartAsTask

    demoBt.Wait()
    //demoTbl.Wait()
    //demoLinmot.Wait()

    //testTbl()

    printfn "All done."
    0 
