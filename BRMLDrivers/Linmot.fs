namespace BRML.Drivers

open System
open System.IO
open System.IO.Ports
open System.Threading


module LinmotIF =

    [<Literal>]
    let Debug = false

    type StateT =
        | NotReadyToSwitchOn
        | ReadyToSwitchOn
        | Error
        | OperationEnabled
        | Homing
        | UnknownState

    type DefaultResponse = {
        // status word
        OpEnabled:              bool;
        Error:                  bool;
        FatalError:             bool;
        InTargetPos:            bool;
        Homed:                  bool;
        MotionActive:           bool;    

        Pos:                    float;
    
        // state var
        State:                  StateT;
        Finished:               bool;
        MotionCmdCnt:           int option;
    }

    type WarnResponse = {
        PositionLag:            bool;
    }

    type Control = {
        SwitchOn:               bool;
        EnableOp:               bool;
        ErrorAck:               bool;
        Home:                   bool;
    }

    type LinmotT = {
        Port:                   SerialPort;
        Id:                     int;
        mutable MotionCmdCnt:   int;       
    }


    let IdleCntrl = {
        SwitchOn =              true;
        EnableOp =              true;
        ErrorAck =              false;
        Home =                  false;
    }

    let private isBitSet bit (x: uint32) = 
        x &&& (1u <<< bit) <> 0u

    let setBit bit value (x: uint32) =
        if value then x ||| (1u <<< bit)
        else x &&& ~~~(1u <<< bit)

    let dumpMsg msg = 
        for b in msg do
            printf "%02x " (int b)

    let sendMsg dataGen linmot =
        if Debug then printf "TX "

        // generate data
        use dataMs = new MemoryStream()       
        use dataBw = new BinaryWriter (dataMs)       
        let (mainId: byte), (subId: byte) = dataGen dataBw linmot
        dataBw.Flush ()
        let dataBuf = dataMs.ToArray()

        // create msg with header and footer
        use msgMs = new MemoryStream()
        use w = new BinaryWriter(msgMs)
        w.Write (byte 0x01)
        w.Write (byte (linmot.Id))
        w.Write (byte (Array.length dataBuf + 3))
        w.Write (byte 0x02)
        w.Write (byte subId)
        w.Write (byte mainId)
        w.Write (dataBuf)
        w.Write (byte 0x04)
        w.Flush()

        // send msg
        let msg = msgMs.ToArray()
        if Debug then printf ": "; dumpMsg msg; printfn ""
        linmot.Port.Write(msg, 0, msg.Length)

    let genResponseReqWithWarnRequest w linmot =
        if Debug then printf "ResponseReqWithWarnRequest"
        byte 0x00, byte 0x03

    let genWriteControl (control: Control) (w: BinaryWriter) linmot =
        if Debug then printf "WriteControl %A" control
        let word =
            0u
            |> setBit 0 control.SwitchOn
            |> setBit 2 true  // not quick stop
            |> setBit 3 control.EnableOp
            |> setBit 4 true  // not abort
            |> setBit 5 true  // not freeze
            |> setBit 7 control.ErrorAck
            |> setBit 11 control.Home
        w.Write (uint16 word)
        byte 0x01, byte 0x00

    let genResetDrive w linmot =
        byte 0x06, byte 0x01
    
    let genMotionCmd cmd (w: BinaryWriter) linmot =
        if Debug then printf "MotionCmd (%02x)" cmd
        linmot.MotionCmdCnt <- linmot.MotionCmdCnt + 1
        let word = (cmd &&& 0xfff0) ||| (linmot.MotionCmdCnt &&& 0x000f)
        linmot.MotionCmdCnt <- linmot.MotionCmdCnt + 1
        if Debug then printf "MotionCmdWord (%04x)" word
        w.Write (uint16 word)
        byte 0x02, byte 0x00

    let genGoToPos pos vel accel decel w linmot =
        if Debug then printf "GoToPos (%f, %f, %f, %f)" pos vel accel decel
        let ids = genMotionCmd 0x0100 w linmot
        w.Write (int32  (pos * 1E+4))
        w.Write (uint32 (vel * 1E+3))
        w.Write (uint32 (accel * 1E+2))
        w.Write (uint32 (decel * 1E+2))
        ids


    let readFromPort count linmot =
        let buf: byte[] = Array.zeroCreate count
        let mutable bytesRead = 0
        while bytesRead < count do
            let readCount = linmot.Port.Read(buf, bytesRead, count - bytesRead)
            bytesRead <- bytesRead + readCount
        buf


    let recvResp dataParser linmot =             
        // read and parse header
        let buf = readFromPort 6 linmot
        if Debug then printf "RX: "; dumpMsg buf
        if buf.[0] <> (byte 0x01) then failwith "received invalid header start byte"
        if (int buf.[1]) <> linmot.Id then failwith "received wrong linmot id"
        let length = int buf.[2]
        if buf.[3] <> (byte 0x02) then failwith "received invalid data start byte"
        let subId = buf.[4]
        let mainId = buf.[5]

        // read payload data
        let buf = readFromPort (length - 2) linmot
        if Debug then dumpMsg buf

        // parse
        use ms = new MemoryStream (buf)
        use r = new BinaryReader (ms, Text.Encoding.ASCII, true)
        let data = dataParser r (mainId, subId) linmot

        if buf.[buf.Length - 1] <> (byte 0x04) then failwith "received invalid end byte" 
        if Debug then printfn ""

        data

    let requireId (id: byte * byte) required =
        if id <> required then 
            failwithf "expected message with id %A but received id %A" required id

    let parseDefaultResponse (r: BinaryReader) id linmot =
        requireId id (byte 0x00, byte 0x00)

        let commState = r.ReadByte()
        let statusWord = r.ReadUInt16()
        let stateWord = r.ReadUInt16()
        let pos = (float (r.ReadInt32())) / 1E+4

        let mainState = ((int stateWord) >>> 8) &&& 0xff
        let subState = (int stateWord) &&& 0xff
        let state =
            match mainState with
            | ms when ms = 0x00 -> NotReadyToSwitchOn
            | ms when ms = 0x02 -> ReadyToSwitchOn
            | ms when ms = 0x04 -> Error
            | ms when ms = 0x08 -> OperationEnabled
            | ms when ms = 0x09 -> Homing
            | _ -> UnknownState
        let motionCmdCount =
            if state = OperationEnabled then Some (subState &&& 0x0f)
            else None

        let sBit b = isBitSet b (uint32 statusWord)
        let status = {
            OpEnabled   = sBit 0;
            Error              = sBit 3;
            FatalError         = sBit 12;
            InTargetPos        = sBit 10;
            Homed              = sBit 11;
            MotionActive       = sBit 13;
            Pos                = pos;
            State              = state;
            Finished           = subState = 0x0f;
            MotionCmdCnt       = motionCmdCount;
        }

        if Debug then printf "DefaultResponse %A" status
        status

    let parseDefaultResponseWithWarnWord (r: BinaryReader) id linmot =        
        let status = parseDefaultResponse r id linmot

        let warnWord = r.ReadUInt16()
        let sBit b = isBitSet b (uint32 warnWord)
        let warn = {
            PositionLag        = sBit 4;
        }

        if Debug then printf "WithWarnWord %A" warn 
        status, warn

    let getStatus linmot =
        sendMsg (genResponseReqWithWarnRequest) linmot
        recvResp (parseDefaultResponseWithWarnWord) linmot

    let sendAndRecvAck dataGen linmot =
        sendMsg dataGen linmot
        recvResp parseDefaultResponse linmot |> ignore
   
      
[<AutoOpen>]
module Linmot =  
    open LinmotIF

    type LinmotCfgT = {
        PortName:           string;
        PortBaud:           int;
        Id:                 int;    
        DefaultVel:         float;
        DefaultAccel:       float;
    }

    type private MsgT =
        | MsgInit
        | MsgHome of bool
        | MsgDriveTo of float * float * float * float
        | MsgPower of bool

    type private ReplyT =
        | ReplyOk
        | ReplyNotHomed
        | ReplyOutOfRange

    type private MsgWithReplyT = MsgT * AsyncReplyChannel<ReplyT>

    type LinmotT (config: LinmotCfgT) =

        let linmot = {
            LinmotT.Port  = new SerialPort(config.PortName, config.PortBaud);
            Id            = config.Id;
            MotionCmdCnt  = 0;
        }
        do linmot.Port.Open()

        let statusUpdatedEventInt = new Event<_>()
        let statusUpdatedEvent = statusUpdatedEventInt.Publish

        let mutable statusThreadShouldRun = true

        let mutable currentStatus = getStatus linmot

        let statusThread = Thread(fun () -> 
            while statusThreadShouldRun do
                lock linmot (fun () ->
                    currentStatus <- getStatus linmot                    
                    statusUpdatedEventInt.Trigger currentStatus
                )                
                Thread.Yield () |> ignore
                if Debug then Thread.Sleep (1000) |> ignore
            )

        let rec waitForState cond = async {
            let! sts, _ = Async.AwaitEvent statusUpdatedEvent
            if not (cond sts) then do! waitForState cond
        }
            
        let agent =
            MailboxProcessor<MsgWithReplyT>.Start(fun inbox ->
                async { 
                    while true do
                        let! msg, rc = inbox.Receive()                       
                        match msg with
                        | MsgInit ->
                            statusThread.Start()
                            let sts, _ = currentStatus
                            if not sts.OpEnabled then
                                if Debug then printfn "Performing ErrorAck"
                                lock linmot (fun() -> 
                                    sendAndRecvAck (genWriteControl {IdleCntrl with SwitchOn=false; EnableOp=false; ErrorAck=true; }) linmot)
                                do! waitForState (fun s -> s.State = ReadyToSwitchOn)
                                if Debug then printfn "Enabling operation"
                                lock linmot (fun() ->
                                    sendAndRecvAck (genWriteControl IdleCntrl) linmot)
                                do! waitForState (fun s -> s.State = OperationEnabled)
                            // update command count nibble
                            let sts, _ = currentStatus
                            linmot.MotionCmdCnt <- sts.MotionCmdCnt.Value

                            rc.Reply ReplyOk

                        | MsgHome force ->
                            let sts, _ = currentStatus
                            if not sts.Homed || force then
                                lock linmot (fun() -> 
                                    sendAndRecvAck (genWriteControl {IdleCntrl with Home=true;}) linmot)
                                do! waitForState (fun s -> s.State = Homing && s.Finished)
                                lock linmot (fun() ->
                                    sendAndRecvAck (genWriteControl IdleCntrl) linmot)
                                do! waitForState (fun s -> s.State = OperationEnabled)
                            rc.Reply ReplyOk

                        | MsgDriveTo (pos, vel, accel, decel) ->
                            lock linmot (fun() ->
                                sendAndRecvAck (genGoToPos pos vel accel decel) linmot)
                            do! waitForState (fun s -> s.State = OperationEnabled && not s.MotionActive && s.InTargetPos)
                            rc.Reply ReplyOk

                        | MsgPower pwr ->
                            lock linmot (fun() -> 
                                sendAndRecvAck (genWriteControl {IdleCntrl with SwitchOn=pwr;}) linmot)
                            rc.Reply ReplyOk
                })

        let postMsg msg = async {
            let! reply = agent.PostAndAsyncReply(fun rc -> msg, rc)
            match reply with
            | ReplyOk -> ()
            | ReplyNotHomed -> failwith "Linmot not homed"
            | ReplyOutOfRange -> failwith "Linmot position out of range"
        }    

        // initialize in background
        do postMsg MsgInit |> Async.Start

        member this.Pos = 
            let sts, _ = currentStatus
            sts.Pos

        member this.Home (?force) =     
            let force = defaultArg force false
            postMsg (MsgHome force)

        member this.DriveTo (pos, ?vel, ?accel, ?decel) =
            let vel = defaultArg vel config.DefaultVel
            let accel = defaultArg accel config.DefaultAccel
            let decel = defaultArg decel config.DefaultAccel
            postMsg (MsgDriveTo(pos, vel, accel, decel))

        member this.Power pwr = postMsg (MsgPower pwr)

        interface IDisposable with
            member this.Dispose () =
                statusThreadShouldRun <- false
                statusThread.Join()
                linmot.Port.Dispose()



        
