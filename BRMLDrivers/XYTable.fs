namespace BRML.Drivers

open System
open System.IO
open System.IO.Ports
open System.Threading
open System.Text.RegularExpressions
open System.Diagnostics


/// low-level driver for RS485 Nanotec stepper motor drivers
module Stepper =

    [<Literal>]
    let Debug = false

    type StepperIdT = int

    type ConfigT = {
        Id:             StepperIdT;
        AnglePerStep:   float;
        StepMode:       int;    
        StartVel:       float;
    }

    type StepperT = 
        {Port:           SerialPort;
         Config:         ConfigT;
         mutable Echo:   bool;}

        member this.Id = this.Config.Id
        member this.AnglePerStep = this.Config.AnglePerStep
        member this.StepMode = this.Config.StepMode
        member this.StartVel = this.Config.StartVel

    type StatusT = {
        Ready:          bool;
        ZeroPos:        bool;
        PosErr:         bool;
    }

    type PositioningModeT =
        | Relative
        | Absolute
        | ExternalReferencing
        | Velocity

    type DirectionT =
        | Left
        | Right


    type private MsgT = Msg of string * int option
        
    #if MeasureTiming
    let mutable sendReqs = 0
    let mutable recvReqs = 0
    let mutable sendTime = int64 0
    let mutable recvTime = int64 0
    #endif

    let private sendMsg (Msg (cmd, arg)) (stepper: StepperT) =
        let argStr =
            match arg with
            | Some arg -> sprintf "%+d" arg
            | None -> ""
        let msgStr = sprintf "#%d%s%s\r" stepper.Id cmd argStr
        if Debug then printfn "sending: %s" msgStr

        #if MeasureTiming 
        let sw = Stopwatch.StartNew()
        stepper.Port.Write msgStr
        sendTime <- sendTime + sw.ElapsedMilliseconds
        sendReqs <- sendReqs + 1
        #else
        stepper.Port.Write msgStr
        #endif

    let private receiveMsg (stepper: StepperT) = 
        if not stepper.Echo then
            printfn "cannot receive message when echo is disabled"
            exit 1

        #if MeasureTiming
        let sw = Stopwatch.StartNew()
        let msgStr = stepper.Port.ReadTo "\r"
        recvTime <- recvTime + sw.ElapsedMilliseconds
        recvReqs <- recvReqs + 1
        #else
        let msgStr = 
            try stepper.Port.ReadTo "\r"
            with :? TimeoutException ->
                printfn "timeout while communicating with XY table"
                exit 1
        #endif


        if Debug then printfn "received: %s" msgStr            
        let m = Regex.Match(msgStr, @"^(\d+)([!-*,./:-~]+)([+-]?\d+)?$")
        if not m.Success then
            failwithf "Stepper received malformed reply message: %s" msgStr
        let id = int m.Groups.[1].Value
        let cmd = m.Groups.[2].Value
        let arg = 
            if m.Groups.Count > 3 && String.length m.Groups.[3].Value > 0 then Some (int m.Groups.[3].Value)
            else None
        if id <> stepper.Id then
            failwithf "Stepper received reply with mismatching id: %s" msgStr

        Msg (cmd, arg)

    let private sendAndReceive msg stepper =
        sendMsg msg stepper
        receiveMsg stepper

    let private sendAndReceiveArg (Msg (msgCmd, _) as msg) stepper =
        let (Msg (replyCmd, replyArg) as reply) = sendAndReceive msg stepper
        if replyCmd <> msgCmd then
            failwithf "Stepper received reply %A with mismatching command for msg %A" reply msg
        match replyArg with
        | Some a -> a
        | None -> failwithf "Stepper received reply %A without argument for msg %A" reply msg
        
    let private sendAndConfirm msg stepper =
        if stepper.Echo then
            let reply = sendAndReceive msg stepper
            if reply <> msg then
                failwithf "Stepper got wrong confirmation %A for message %A" reply msg
        else
            sendMsg msg stepper

    let private isBitSet bit x = x &&& (1 <<< bit) <> 0 

    let private accelToArg accel =
        int ((3000. / ((float accel) / 1000. + 11.7)) ** 2.)

    let private angleToSteps (angle: float) (stepper: StepperT) =
        angle * (float stepper.StepMode) / stepper.AnglePerStep |> int

    let private stepsToAngle (steps: int) (stepper: StepperT) =
        (float steps) * stepper.AnglePerStep / (float stepper.StepMode)

    let getStatus stepper =
        let r = sendAndReceiveArg (Msg ("$", None)) stepper
        {Ready=isBitSet 0 r; ZeroPos=isBitSet 1 r; PosErr=isBitSet 2 r;}

    let isReady stepper =
        let {Ready=ready} = getStatus stepper
        ready

    let setPositioningMode mode stepper =
        let arg =
            match mode with
            | Relative -> 1
            | Absolute -> 2
            | ExternalReferencing -> 4
            | Velocity -> 5
        sendAndConfirm (Msg("p", Some arg)) stepper

    let setDirection direction stepper =
        let arg =
            match direction with
            | Left -> 0
            | Right -> 1
        sendAndConfirm (Msg("d", Some arg)) stepper

    let getPosSteps stepper =               sendAndReceiveArg (Msg ("C", None)) stepper
    let isReferenced stepper =              sendAndReceiveArg (Msg (":is_referenced", None)) stepper = 1
    let setTargetPos pos stepper =          sendAndConfirm (Msg ("s", Some pos)) stepper
    let setStartStepsPerSec sps stepper =   sendAndConfirm (Msg ("u", Some sps)) stepper
    let setDriveStepsPerSec sps stepper =   sendAndConfirm (Msg ("o", Some sps)) stepper
    let setAccel accel stepper =            sendAndConfirm (Msg ("b", Some (accelToArg accel))) stepper
    let setDecel decel stepper =            sendAndConfirm (Msg ("B", Some (accelToArg decel))) stepper
    let setRepetitions reps stepper =       sendAndConfirm (Msg ("W", Some reps)) stepper
    let setFollowCmd cmd stepper =          sendAndConfirm (Msg ("N", Some cmd)) stepper  
    let resetPosErr pos stepper =           sendAndConfirm (Msg ("D", Some pos)) stepper
    let start stepper =                     sendAndConfirm (Msg ("A", None)) stepper
    let stop stepper =                      sendAndConfirm (Msg ("S", Some 1)) stepper
    let quickStop stepper =                 sendAndConfirm (Msg ("S", Some 0)) stepper

    let enableEcho stepper = 
        if not stepper.Echo then
            stepper.Echo <- true
            sendAndConfirm (Msg ("|", Some 1)) stepper
    let disableEcho stepper =
        if stepper.Echo then
            stepper.Echo <- false
            sendAndConfirm (Msg ("|", Some 0)) stepper


    let getPos stepper =
        let p = getPosSteps stepper 
        stepsToAngle p stepper

    let driveTo pos vel accel decel stepper =
        setPositioningMode Absolute stepper
        setTargetPos (angleToSteps pos stepper) stepper
        setStartStepsPerSec (angleToSteps stepper.StartVel stepper) stepper
        setDriveStepsPerSec (angleToSteps vel stepper) stepper
        setAccel (angleToSteps accel stepper) stepper
        setDecel (angleToSteps decel stepper) stepper
        setRepetitions 1 stepper
        setFollowCmd 0 stepper
        start stepper
        
    let splitDirection vel =
        if vel >= 0. then vel, Right
        else -vel, Left

    let startConstantVelocityDrive vel accel decel stepper =
        let absVel, dir = splitDirection vel

        setPositioningMode Velocity stepper
        setDirection dir stepper
        setStartStepsPerSec (angleToSteps stepper.StartVel stepper) stepper
        setDriveStepsPerSec (angleToSteps absVel stepper) stepper
        setAccel (angleToSteps accel stepper) stepper
        setDecel (angleToSteps decel stepper) stepper
        setRepetitions 1 stepper
        setFollowCmd 0 stepper
        start stepper
        
    let adjustVelocity vel signChange stepper =
        let absVel, dir = splitDirection vel
        if signChange then setDirection dir stepper
        setDriveStepsPerSec (angleToSteps absVel stepper) stepper

    let adjustAccelDecel accel decel stepper =
        setAccel (angleToSteps accel stepper) stepper
        setDecel (angleToSteps decel stepper) stepper

    let externalReferencing direction vel accel stepper =
        setPositioningMode ExternalReferencing stepper
        setDirection direction stepper
        setStartStepsPerSec (angleToSteps stepper.StartVel stepper) stepper
        setDriveStepsPerSec (angleToSteps vel stepper) stepper
        setAccel (angleToSteps accel stepper) stepper
        setDecel (angleToSteps accel stepper) stepper
        setRepetitions 1 stepper
        setFollowCmd 0 stepper
        start stepper


module XYTable =

    type AxisConfigT = {
        StepperConfig:  Stepper.ConfigT;
        DegPerMM:       float;
        Home:           Stepper.DirectionT;
        MaxPos:         float;
    }

    type XYTableConfigT = {
        PortName:       string;
        PortBaud:       int;
        X:              AxisConfigT;
        Y:              AxisConfigT;
        DefaultVel:     float;
        DefaultAccel:   float;
        HomeVel:        float;
    }

    type XYTupleT = float * float

    type private MsgT =
        | MsgHome
        | MsgDriveTo of XYTupleT * XYTupleT * XYTupleT * XYTupleT
        | MsgDriveWithVel of XYTupleT * XYTupleT 
        | MsgStop of XYTupleT

    type private ReplyT =
        | ReplyOk
        | ReplyNotHomed
        | ReplyOutOfRange

    type private MsgWithReplyT = MsgT * AsyncReplyChannel<ReplyT>

    type private TrackMode =
        | Query
        | Simulate

    type XYTableT (config: XYTableConfigT) =
        inherit System.Runtime.ConstrainedExecution.CriticalFinalizerObject()

        let port = new SerialPort(config.PortName, config.PortBaud)     
        do
            port.Open()  
            port.ReadTimeout <- 30000
            port.WriteTimeout <- 30000

        let xStepper = {Stepper.StepperT.Port=port; Stepper.StepperT.Config=config.X.StepperConfig; Stepper.StepperT.Echo=false}
        let yStepper = {Stepper.StepperT.Port=port; Stepper.StepperT.Config=config.Y.StepperConfig; Stepper.StepperT.Echo=false}

        do
            Stepper.enableEcho xStepper
            Stepper.enableEcho yStepper

        let mutable disposed = false

        let quickStop () =
            for i = 1 to 10 do
                Stepper.quickStop xStepper
                Stepper.quickStop yStepper
    
        let posInRange (x, y) =
            -0.2 <= x && x <= config.X.MaxPos && -0.2 <= y && y <= config.Y.MaxPos

        let lastPosQuery = Stopwatch.StartNew()
        let lastPosReport = Stopwatch.StartNew()

        [<VolatileField>]
        let mutable currentPos = 0., 0.

        [<VolatileField>]
        let mutable currentVel = 0., 0.

        [<VolatileField>]
        let mutable targetVel = 0., 0.

        [<VolatileField>]
        let mutable currentAccel = 0., 0.

        [<VolatileField>]
        let mutable currentTime = 0.

        [<VolatileField>]
        let mutable trackMode = Query
        
        [<VolatileField>]
        let mutable overshoot = false
        
        [<VolatileField>]
        let mutable homed = false

        [<VolatileField>]
        let mutable waitingForReady = false

        [<VolatileField>]
        let mutable posReportInterval = 10

        [<VolatileField>]
        let mutable sentinelThreadShouldRun = true

        let fetchPos () = 
            let xPos, yPos = Stepper.getPos xStepper, Stepper.getPos yStepper
            let xPos = if config.X.Home = Stepper.Right then -xPos else xPos
            let yPos = if config.Y.Home = Stepper.Right then -yPos else yPos
            xPos / config.X.DegPerMM, yPos / config.Y.DegPerMM

        let fetchStatus () =
            Stepper.getStatus xStepper, Stepper.getStatus yStepper

        let readyEventInt = new Event<_>()
        let readyEvent = readyEventInt.Publish

        let posAcquired = new Event<XYTupleT>()

        let simulateToNow () =
            let maxDt = 0.001

            let stepAxis tVel pos vel accel dt =
                let dVel = tVel - vel
                let vel = 
                    if abs dVel < accel * dt then tVel
                    else vel + accel * (sign dVel |> float) * dt
                let pos = pos + vel * dt
                pos, vel

            lock currentPos (fun () ->
                let toTime = float (Stopwatch.GetTimestamp()) / float Stopwatch.Frequency
                while abs (toTime - currentTime) >= maxDt do
                    let x, y = currentPos
                    let vx, vy = currentVel
                    let ax, ay = currentAccel
                    let tvx, tvy = targetVel
                    let dt = 
                        if toTime - currentTime > maxDt then maxDt
                        else toTime - currentTime
                    let x, vx = stepAxis tvx x vx ax dt
                    let y, vy = stepAxis tvy y vy ay dt
                    currentPos <- x, y
                    currentVel <- vx, vy
                    currentTime <- currentTime + dt
            )

        let startSimulation () =
            lock currentPos (fun () ->
                currentVel <- 0., 0.
                currentAccel <- 0., 0.
                currentTime <- float (Stopwatch.GetTimestamp()) / float Stopwatch.Frequency
                trackMode <- Simulate
            )

        let stopSimulation () =
            lock currentPos (fun () ->
                simulateToNow ()
                trackMode <- Query       
            )

        let setTargetVel vel accel =
            lock currentPos (fun () ->
                simulateToNow ()
                targetVel <- vel
                currentAccel <- accel
            )            
                

        let queryPosAndStatus () =
            Stepper.enableEcho xStepper; Stepper.enableEcho yStepper

            currentPos <- fetchPos ()      
            lastPosQuery.Restart ()             
                    
            if waitingForReady then                       
                let xStatus, yStatus = fetchStatus ()
                if xStatus.Ready && yStatus.Ready then
                    readyEventInt.Trigger()
                if xStatus.PosErr || yStatus.PosErr then
                    printfn "====== XYTable position error"
                    quickStop ()
                    exit -1
                if Stepper.Debug then
                    printfn "X status: %A" xStatus
                    printfn "Y status: %A" yStatus 
                    printfn "Position: %A" currentPos

        let sentinelThread = Thread(fun () -> 
            while sentinelThreadShouldRun do

                // update position
                if trackMode = Simulate then simulateToNow ()               
                let needQuery = 
                    match trackMode with
                    | Query -> true
                    | Simulate -> waitingForReady || lastPosQuery.ElapsedMilliseconds > 200L
                if needQuery then lock port queryPosAndStatus

                // check that position is in range
                if homed && not (posInRange currentPos) then
                    printfn "====== XYTable overshoot: %A" currentPos 
                    overshoot <- true
                    lock port quickStop
                    exit -1                

                // report position
                if lastPosReport.ElapsedMilliseconds >= int64 posReportInterval then
                    async { posAcquired.Trigger currentPos } |> Async.Start
                    lastPosReport.Restart ()

                if trackMode = Query then Thread.Yield() |> ignore
                if Stepper.Debug then Thread.Sleep(500)                           
        )

        do sentinelThread.Start()

        let waitForReady() = async {
            waitingForReady <- true
            do! Async.AwaitEvent readyEvent
            waitingForReady <- false
        }

        let agent =
            MailboxProcessor<MsgWithReplyT>.Start(fun inbox ->
                async { 
                    let xDeg mm = config.X.DegPerMM * mm
                    let yDeg mm = config.Y.DegPerMM * mm

                    // status variables
                    let vmActive = ref false
                    let vmVel = ref (0., 0.)
                    let vmAccel = ref (0., 0.)
                    let lastRequest = Stopwatch.StartNew()

                    while true do
                        let! msg, rc = inbox.Receive()    
                        
                        // sending commands at a too high rate will hang the motor controller
                        while not (lastRequest.ElapsedMilliseconds > 10L) do ()
                                           
                        match msg with
                        | MsgHome ->
                            if not homed then 
                                lock port (fun () ->
                                    Stepper.enableEcho xStepper; Stepper.enableEcho yStepper

                                    let {Stepper.PosErr=xPosErr}, {Stepper.PosErr=yPosErr} = fetchStatus ()
                                    let xHomed, yHomed = Stepper.isReferenced xStepper, Stepper.isReferenced yStepper
                                    let pos = fetchPos ()

                                    if (not xHomed) || (not yHomed) || xPosErr || yPosErr || (not (posInRange pos)) then
                                        Stepper.resetPosErr 1000 xStepper
                                        Stepper.resetPosErr 1000 yStepper
                                        Stepper.externalReferencing config.X.Home (xDeg config.HomeVel) 
                                            (xDeg config.DefaultAccel) xStepper
                                        Stepper.externalReferencing config.Y.Home (yDeg config.HomeVel) 
                                            (yDeg config.DefaultAccel) yStepper
                                )                                                       
                                do! waitForReady()
                                do! Async.Sleep 100
                                homed <- true
                            rc.Reply ReplyOk

                        | MsgDriveTo (((xpos, ypos) as pos), (xvel, yvel), (xaccel, yaccel), (xdecel, ydecel)) ->
                            if not (posInRange pos) then rc.Reply ReplyOutOfRange
                            if not homed then rc.Reply ReplyNotHomed
                            else 
                                let xpos = if config.X.Home = Stepper.Right then -xpos else xpos
                                let ypos = if config.Y.Home = Stepper.Right then -ypos else ypos
                                lock port (fun () ->
                                    Stepper.enableEcho xStepper; Stepper.disableEcho yStepper                                        
                                    Stepper.driveTo (xDeg xpos) (xDeg xvel) (xDeg xaccel) (xDeg xdecel) xStepper
                                    Stepper.driveTo (yDeg ypos) (yDeg yvel) (yDeg yaccel) (yDeg ydecel) yStepper
                                )
                                do! waitForReady()
                                rc.Reply ReplyOk

                        | MsgDriveWithVel ((xvel, yvel as vel), (xaccel, yaccel as accel)) ->
                            if not homed then rc.Reply ReplyNotHomed
                            else
                                let minVel = 0.001
                                let cntrlXvel = if config.X.Home = Stepper.Right then -xvel else xvel
                                let cntrlYvel = if config.Y.Home = Stepper.Right then -yvel else yvel
                                let cntrlXvel = if abs cntrlXvel < minVel then minVel else cntrlXvel
                                let cntrlYvel = if abs cntrlYvel < minVel then minVel else cntrlYvel
                                let cntrlVel = cntrlXvel, cntrlYvel
                                lock port (fun () ->
                                    if not !vmActive then
                                        Stepper.startConstantVelocityDrive (xDeg cntrlXvel) (xDeg xaccel) (xDeg xaccel) xStepper
                                        Stepper.startConstantVelocityDrive (yDeg cntrlYvel) (yDeg yaccel) (yDeg yaccel) yStepper
                                        vmActive := true
                                        vmVel := cntrlVel
                                        vmAccel := accel

                                        startSimulation ()
                                        setTargetVel vel accel
                                    else
                                        Stepper.disableEcho xStepper; Stepper.disableEcho yStepper                                        
                                        if accel <> !vmAccel then
                                            Stepper.adjustAccelDecel (xDeg xaccel) (xDeg xaccel) xStepper
                                            Stepper.adjustAccelDecel (yDeg yaccel) (yDeg yaccel) yStepper
                                            vmAccel := accel
                                        if cntrlVel <> !vmVel then
                                            let vmXVel, vmYVel = !vmVel
                                            Stepper.adjustVelocity (xDeg cntrlXvel) (sign xvel <> sign vmXVel) xStepper
                                            Stepper.adjustVelocity (yDeg cntrlYvel) (sign yvel <> sign vmYVel) yStepper
                                            vmVel := cntrlVel

                                        setTargetVel vel accel
                                )
                                rc.Reply ReplyOk

                        | MsgStop (xaccel, yaccel as accel) ->      
                            lock port (fun () ->     
                                Stepper.enableEcho xStepper; Stepper.disableEcho yStepper                                        
                                if accel <> !vmAccel then
                                    Stepper.adjustAccelDecel (xDeg xaccel) (xDeg xaccel) xStepper
                                    Stepper.adjustAccelDecel (yDeg yaccel) (yDeg yaccel) yStepper
                                    vmAccel := accel
                                Stepper.stop xStepper
                                Stepper.stop yStepper
                                vmActive := false

                                stopSimulation ()
                            )
                            do! waitForReady()
                            rc.Reply ReplyOk

                        lastRequest.Restart()
                }           
            )

        let postMsg msg = async {
            let! reply = agent.PostAndAsyncReply(fun rc -> msg, rc)
            match reply with
            | ReplyOk -> ()
            | ReplyNotHomed -> failwith "XYTable not homed"
            | ReplyOutOfRange -> failwith "XYTable position out of range"
        }

        let terminate () =
            //printfn "Terminate"
            if not disposed then
                try
                    sentinelThreadShouldRun <- false         
                    if sentinelThread <> null && sentinelThread.IsAlive then sentinelThread.Join ()
                finally
                    if port.IsOpen then quickStop ()                     

        do
            AppDomain.CurrentDomain.ProcessExit.Add(fun _ -> terminate())
            AppDomain.CurrentDomain.UnhandledException.Add(fun _ -> terminate())
            AppDomain.CurrentDomain.DomainUnload.Add(fun _ -> terminate())
            Console.CancelKeyPress.Add(fun _ -> terminate())

        member this.CurrentPos = 
            if trackMode = Simulate then simulateToNow ()
            currentPos

        member this.PosAcquired = posAcquired.Publish

        member this.GetNextPos () =
            Async.AwaitEvent (this.PosAcquired) |> Async.RunSynchronously
        
        member this.Home() = 
            postMsg (MsgHome)

        member this.DriveTo (pos, ?vel, ?accel, ?decel) = 
            let vel = defaultArg vel (config.DefaultVel, config.DefaultVel)
            let accel = defaultArg accel (config.DefaultAccel, config.DefaultAccel)
            let decel = defaultArg decel (config.DefaultAccel, config.DefaultAccel)
            postMsg (MsgDriveTo (pos, vel, accel, decel))

        member this.DriveWithVel (vel, ?accel) = 
            let accel = defaultArg accel (config.DefaultAccel, config.DefaultAccel)
            postMsg (MsgDriveWithVel (vel, accel)) |> Async.RunSynchronously

        member this.Stop (?accel) = 
            let accel = defaultArg accel (config.DefaultAccel, config.DefaultAccel)       
            postMsg (MsgStop (accel)) |> Async.RunSynchronously

        member this.PosReportInterval 
            with get () = posReportInterval
            and set value = 
                if 0 <= value then posReportInterval <- value
                else invalidArg "value" "PosReportInterval must be >= 0 ms"

        interface IDisposable with
            member this.Dispose () =
                terminate ()
                port.Dispose()
                disposed <- true

        override this.Finalize() =
            terminate ()

        interface Datasets.RecorderTypes.ISensor<XYTupleT> with
            member this.DataType = typeof<XYTupleT>
            member this.SampleAcquired = this.PosAcquired
            member this.Interpolate fac a b =
                let ax, ay = a
                let bx, by = b
                let x = (1.-fac) * ax + fac * bx
                let y = (1.-fac) * ay + fac * by
                x, y



[<AutoOpen>]
module XYTableTypes =
    type XYTableCfgT = XYTable.XYTableConfigT
    type XYTableT = XYTable.XYTableT




