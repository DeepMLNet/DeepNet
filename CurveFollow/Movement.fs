module Movement

open System
open System.IO
open RProvider
open RProvider.graphics
open RProvider.grDevices
open Nessos.FsPickler
open Nessos.FsPickler.Json

open Basics
open ArrayNDNS


type XY = float * float

/// Braille coordinate system.
module BrailleCS =

    let dot_radius = 0.72       // mm
    let dot_height = 0.48       // mm
    let dot_dist = 2.34         // mm
    let char_dist = 6.2         // mm
    let line_dist = 10.0        // mm
    let x_offset = 22.67        // mm
    let y_offset = 18.5         // mm
    let hole1_x = 6.5           // mm
    let hole1_y = 31.0          // mm
    let hole1_radius = 3.2      // mm
    let hole2_x = 7.5           // mm
    let hole2_y = 111.5         // mm
    let hole2_radius = 3.3      // mm
    let label_x = 4.0           // mm
    let label_y = 42.0          // mm
    let label_thickness = 1.0   // mm
    let page_width = 206.0      // mm
    let page_height = 145.0     // mm
    let page_thickness = 0.4    // mm
    let clip_width = 10.0       // mm
    let clip_height = 1.0       // mm
    let clip_thickness = 6.0    // mm
    //let col_offset = 3.0        // chars
    let col_offset = -3.0        // chars
    let row_offset = 0.0        // lines

    /// Relative position of dot (0..5) to character in mm.
    let dotRelMM (dot: int) =
        assert (0 <= dot && dot < 6)
        let x = dot_dist * float (dot / 3) - dot_dist / 2.
        let y = dot_dist * float (dot % 3) - dot_dist
        x, y

    /// Braille character index to table position in mm.
    let charToMM (col: float, row: float) = 
        let x = char_dist * (col + col_offset) + x_offset
        let y = line_dist * (row + row_offset) + y_offset
        x, y

    /// Braille character index and dot number to table position in mm.
    let dotToMM (col, row) dot =
        let xc, yc = charToMM (col, row)
        let xd, yd = dotRelMM dot
        xc + xd, yc + yd


type DistortionCfg = {
    DistortionsPerSec:      float
    MaxOffset:              float
    MinHold:                float
    MaxHold:                float
    NotAgainFor:            float
}

type private DistortionState = 
    | InactiveUntil of float
    | GotoPos of float
    | HoldUntil of float

type Mode =
    | FixedOffset of float
    | Distortions of DistortionCfg

type Cfg = {
    Dt:             float
    Accel:          float
    VelX:           float
    MaxVel:         float
    MaxControlVel:  float
    Mode:           Mode
    IndentorPos:    float
}

type MovementPoint = {
    Time:           float
    Pos:            XY
    ControlVel:     XY
    CurveY:         float
    Distorted:      bool
}

type Movement = {
    StartPos:       XY
    IndentorPos:    float
    Accel:          float
    VelX:           float
    Points:         MovementPoint list
}

type RecordedMovementPoint = {
    Time:           float
    SimPos:         XY
    DrivenPos:      XY
    ControlVel:     XY
    YDist:          float
    Distorted:      bool
    Biotac:         float []
}

type RecordedMovement = {
    IndentorPos:    float
    Accel:          float
    Points:         RecordedMovementPoint list
}


let generate (cfg: Cfg) (rnd: System.Random) (curve: XY list) = 
    let tblCfg = {
        XYTableSim.Accel  = cfg.Accel, cfg.Accel 
        XYTableSim.Dt     = cfg.Dt
        XYTableSim.MaxVel = cfg.MaxVel, cfg.MaxVel
    }

    let _, baseY = curve.[0]
    let startPos = 
        match cfg.Mode with
        | FixedOffset offset -> let x, y = curve.[0] in x, y + offset
        | _ -> curve.[0]   

    let controlPid = PID.Controller {
        PID.PFactor     = 4.0
        PID.IFactor     = 2.0
        PID.DFactor     = 1.0
        PID.ITime       = 0.05
        PID.DTime       = 0.05
    }
    let maxOffset = 9.6

    let rec generate curve (state: XYTableSim.State) distState = seq {
        let movementPoint cVel cy = {
            Time        = state.Time
            Pos         = state.Pos
            ControlVel  = cVel
            CurveY      = cy
            Distorted   = match distState with InactiveUntil _ -> true | _ -> false
        }
        let x, y = state.Pos
        let t = state.Time
        match curve with
        | [] -> ()
        | (x1,y1) :: (x2,y2) :: _ when x1 <= x && x < x2 ->
            // interpolate curve points
            let fac = (x2 - x) / (x2 - x1)
            let cy = fac * y1 + (1. - fac) * y2

            match cfg.Mode with
            | FixedOffset ofst ->         
                let trgt = cy + ofst
                let trgt = min trgt (baseY + maxOffset)
                let trgt = max trgt (baseY - maxOffset)
                let cVel = cfg.VelX, controlPid.Simulate trgt y t
                yield movementPoint cVel cy
                yield! generate curve (XYTableSim.step cVel tblCfg state) distState
            | Distortions dc ->
                let cPos, nextState =
                    match distState with
                    | InactiveUntil iu ->               
                        let prob = dc.DistortionsPerSec * cfg.Dt
                        if x >= iu && rnd.NextDouble() < prob then
                            let trgt = cy + (2.0 * rnd.NextDouble() - 1.0) * dc.MaxOffset
                            let trgt = min trgt (baseY + maxOffset)
                            let trgt = max trgt (baseY - maxOffset)
                            cy, GotoPos trgt
                        else cy, distState
                    | GotoPos trgt ->
                        if abs (y - trgt) < 0.05 then
                            let hu = x + dc.MinHold + rnd.NextDouble() * (dc.MaxHold - dc.MinHold)
                            trgt, HoldUntil hu
                        else trgt, distState
                    | HoldUntil hu ->
                        if x >= hu then cy, InactiveUntil (x + dc.NotAgainFor)
                        else y, distState

                let cVel = cfg.VelX, controlPid.Simulate cPos y t               
                yield movementPoint cVel cy
                yield! generate curve (XYTableSim.step cVel tblCfg state) nextState

        | (x1,y1) :: _ when x < x1 ->
            // x position is left of curve start
            let vel = cfg.VelX, 0.
            yield movementPoint vel y1
            yield! generate curve (XYTableSim.step vel tblCfg state) distState
        | _ :: rCurve ->
            // move forward on curve
            yield! generate rCurve state distState
    }

    let state = {XYTableSim.Time=0.; XYTableSim.Pos=startPos; XYTableSim.Vel=0., 0. }
    let movement = generate curve state (InactiveUntil 0.3) |> Seq.toList 

    {
        StartPos    = startPos
        IndentorPos = cfg.IndentorPos
        Accel       = cfg.Accel
        VelX        = cfg.VelX
        Points      = movement
    }


let toDriveCurve (movement: Movement) = 
    {
        TactileCurve.IndentorPos = movement.IndentorPos
        TactileCurve.StartPos    = movement.StartPos
        TactileCurve.Accel       = movement.Accel
        TactileCurve.XVel        = movement.VelX
        TactileCurve.Points      = [ for mp in movement.Points -> 
                                     {
                                        TactileCurve.XPos = fst mp.Pos
                                        TactileCurve.YPos = snd mp.Pos
                                     } ]            
    }


let syncTactileCurve (tc: TactileCurve.TactileCurve) (m: Movement) =
    let rec syncPoints (tcPoints: TactileCurve.TactilePoint list) (mPoints: MovementPoint list) synced = 
        //printfn "tcPoints: %A   mPoints: %A" (List.head tcPoints) (List.head mPoints) 
        match tcPoints, mPoints with
        | [], _  
        | _, [] -> List.rev synced
        | ({Time=tct} as t)::tcRest, ({Time=mt} as m)::({Time=mtNext} as mNext)::_ when mt <= tct && tct < mtNext ->
            let fac = float (mtNext - tct) / float (mtNext - mt)
            let interp a b = 
                let xa, ya = a
                let xb, yb = b
                let x = fac * xa + (1.0 - fac) * xb
                let y = fac * ya + (1.0 - fac) * yb
                x, y
            let rp = {
                Time       = tct
                SimPos     = interp m.Pos mNext.Pos
                DrivenPos  = t.Pos
                ControlVel = interp m.ControlVel mNext.ControlVel
                YDist      = m.CurveY - snd t.Pos
                Distorted  = m.Distorted
                Biotac     = t.Biotac
            }
            syncPoints tcRest mPoints (rp::synced)
        | {Time=tct}::tcRest, {Time=mt}::_ when tct < mt ->
            syncPoints tcRest mPoints synced
        | _, _::mRest ->
            syncPoints tcPoints mRest synced
       
    let synced = syncPoints tc.Points m.Points []

    {
        IndentorPos = tc.IndentorPos
        Accel       = tc.Accel
        Points      = synced
    }

let loadCurves path =
    use file = NPZFile.Open path
    let pos: ArrayNDHostT<float> = file.Get "pos" // pos[dim, idx, smpl]
    seq { for smpl = 0 to pos.Shape.[2] - 1 do
              yield [for idx = 0 to pos.Shape.[1] - 1 do
                         // convert to mm
                         let col, row = pos.[[0; idx; smpl]], pos.[[1; idx; smpl]]
                         yield BrailleCS.charToMM (col, row) ] }  
    |> List.ofSeq
    
let toArray extract points = 
    let xAry = Array.zeroCreate (List.length points)
    let yAry = Array.zeroCreate (List.length points)
    for idx, pnt in List.indexed points do
        let x, y = extract pnt
        xAry.[idx] <- x
        yAry.[idx] <- y
    xAry, yAry
    

let plotMovement (path: string) (curve: XY list) (movement: Movement) =
    let curveX, curveY = toArray id curve
    let posX, posY = toArray (fun (p: MovementPoint) -> p.Pos) movement.Points
    let controlVelX, controlVelY = toArray (fun (p: MovementPoint) -> p.ControlVel) movement.Points
    let distorted = movement.Points |> List.map (fun p -> p.Distorted) |> Array.ofList

    R.pdf (path) |> ignore
    R.par2 ("oma", [0; 0; 0; 0])
    R.par2 ("mar", [3.2; 2.6; 1.0; 0.5])
    R.par2 ("mgp", [1.7; 0.7; 0.0])
    R.par2 ("mfrow", [2; 1])

    R.plot2 ([0; 150], [curveY.[0] - 10.; curveY.[0] + 10.], "position", "x", "y")
    R.abline(h=curveY.[0]) |> ignore
    R.lines2 (curveX, curveY, "black")
    R.lines2 (posX, posY, "blue")
    R.legend (115., curveY.[0] + 10., ["curve"; "movement"], col=["black"; "blue"], lty=[1;1]) |> ignore

    R.plot2 ([0; 150], [-20; 20], "velocity", "x", "y velocity")
    R.abline(h=0) |> ignore
    R.lines2 (posX, controlVelY, "blue")
    R.legend (125., 20, ["control"], col=["blue"], lty=[1;1]) |> ignore

    R.dev_off() |> ignore


let plotTactile (path: string) (curve: XY list) (tactile: TactileCurve.TactileCurve) =
    let curveX, curveY = toArray id curve
    let drivenPosX, drivenPosY = tactile.Points |> toArray (fun p -> p.Pos) 
    let biotac = tactile.Points |> List.map (fun p -> p.Biotac)

    let dt = tactile.Points.[1].Time - tactile.Points.[0].Time
    let drivenVelY =
        drivenPosY
        |> Array.toSeq
        |> Seq.pairwise
        |> Seq.map (fun (a, b) -> (b - a) / dt)
        |> Seq.append (Seq.singleton 0.)
        |> Array.ofSeq

    R.pdf (path) |> ignore
    R.par2 ("oma", [0; 0; 0; 0])
    R.par2 ("mar", [3.2; 2.6; 1.0; 0.5])
    R.par2 ("mgp", [1.7; 0.7; 0.0])
    R.par2 ("mfrow", [2; 1])

    R.plot2 ([0; 150], [curveY.[0] - 6.; curveY.[0] + 6.], "position", "x", "y")
    R.abline(h=curveY.[0]) |> ignore
    R.lines2 (curveX, curveY, "black")
    R.lines2 (drivenPosX, drivenPosY, "yellow")
    R.legend (115., curveY.[0] + 6., ["curve"; "driven"], col=["black"; "yellow"], lty=[1;1]) |> ignore

    R.plot2 ([0; 150], [-20; 20], "velocity", "x", "y velocity")
    R.abline(h=0) |> ignore
    R.lines2 (drivenPosX, drivenVelY, "yellow")
    R.legend (125., 20, ["driven"], col=["yellow"], lty=[1;1]) |> ignore

    R.dev_off() |> ignore


let plotRecordedMovement (path: string) (curve: XY list) (recMovement: RecordedMovement) (predDistY: float list option) =
    let curveX, curveY = toArray id curve
    let simPosX, simPosY = recMovement.Points |> toArray (fun p -> p.SimPos) 
    let drivenPosX, drivenPosY = recMovement.Points |> toArray (fun p -> p.DrivenPos) 
    let controlVelX, controlVelY = recMovement.Points |> toArray (fun p -> p.ControlVel) 
    let distY = recMovement.Points |> List.map (fun p -> p.YDist) |> Array.ofList
    let distorted = recMovement.Points |> List.map (fun p -> p.Distorted) |> Array.ofList
    let biotac = recMovement.Points |> List.map (fun p -> p.Biotac)

    let left, right = Array.min drivenPosX, Array.max drivenPosX

    let dt = recMovement.Points.[1].Time - recMovement.Points.[0].Time
    let drivenVelY =
        drivenPosY
        |> Array.toSeq
        |> Seq.pairwise
        |> Seq.map (fun (a, b) -> (b - a) / dt)
        |> Seq.append (Seq.singleton 0.)
        |> Array.ofSeq

    R.pdf (path) |> ignore
    R.par2 ("oma", [0; 0; 0; 0])
    R.par2 ("mar", [3.2; 2.6; 1.0; 0.5])
    R.par2 ("mgp", [1.7; 0.7; 0.0])
    R.par2 ("mfrow", [4; 1])

    R.plot2 ([left; right], [curveY.[0] - 10.; curveY.[0] + 10.], "position", "x", "y")
    R.abline(h=curveY.[0]) |> ignore
    R.lines2 (curveX, curveY, "black")
    R.lines2 (simPosX, simPosY, "blue")
    R.lines2 (drivenPosX, drivenPosY, "yellow")
    R.legend (125., curveY.[0] + 10., ["curve"; "movement"; "driven"], col=["black"; "blue"; "yellow"], lty=[1;1]) |> ignore

    R.plot2 ([left; right], [-15; 15], "velocity", "x", "y velocity")
    R.abline(h=0) |> ignore
    R.lines2 (drivenPosX, controlVelY, "blue")
    R.lines2 (drivenPosX, drivenVelY, "yellow")
    R.legend (125., 15, ["control"; "driven"], col=["blue"; "yellow"], lty=[1;1]) |> ignore

    R.plot2 ([left; right], [-8; 8], "distance to curve", "x", "y distance")
    R.abline(h=0) |> ignore
    R.lines2 (drivenPosX, distY, "blue")
    match predDistY with
    | Some predDistY -> 
        R.lines2 (drivenPosX, predDistY, "red")
        R.legend (125., 8, ["true"; "predicted"], col=["blue"; "red"], lty=[1;1]) |> ignore
    | None -> ()

    // plot biotac
    let biotacImg = array2D biotac |> ArrayNDHost.ofArray2D |> ArrayND.transpose  // [chnl, smpl]
    let minVal, maxVal = ArrayND.minAxis 1 biotacImg, ArrayND.maxAxis 1 biotacImg
    let scaledImg = (biotacImg - minVal.[*, NewAxis]) / (maxVal - minVal).[*, NewAxis]
    R.image2 (ArrayNDHost.toArray2D scaledImg, lim=(0.0, 1.0),
              xlim=(left, right), colormap=Gray, title="biotac", xlabel="x", ylabel="channel")

    R.dev_off() |> ignore



let generateMovementForFile cfgs path outDir =
    let rnd = Random ()
    let baseName = Path.Combine(Path.GetDirectoryName path, Path.GetFileNameWithoutExtension path)
    let curves = loadCurves path
    use curveHdf = HDF5.OpenWrite (baseName + ".h5")

    for curveIdx, curve in List.indexed curves do      
        // write curve to HDF5 file
        let ary = ArrayNDHost.zeros [List.length curve; 2]
        for idx, (x, y) in List.indexed curve do
            ary.[[idx; 0]] <- x
            ary.[[idx; 1]] <- y
        ArrayNDHDF.write curveHdf (sprintf "curve%d" curveIdx) ary

        // generate movements
        for cfgIdx, cfg in List.indexed cfgs do
            let dir = Path.Combine(outDir, sprintf "Curve%dCfg%d" curveIdx cfgIdx)
            Directory.CreateDirectory dir |> ignore

            let movement = generate cfg rnd curve
            plotMovement (Path.Combine (dir, "movement.pdf")) curve movement

            if curveIdx <> 0 && curveIdx <> 6 then
                let p = FsPickler.CreateBinarySerializer()
                use tw = File.OpenWrite(Path.Combine (dir, "movement.dat"))
                p.Serialize(tw, movement)
                use tw = File.OpenWrite(Path.Combine (dir, "curve.dat"))
                p.Serialize(tw, curve)
            
type GenCfg = {
    CurveDir:           string
    MovementDir:        string
    MovementCfgs:       Cfg list
}

let generateMovementUsingCfg cfg  =
    for file in Directory.EnumerateFiles(cfg.CurveDir, "*.cur.npz") do
        printfn "%s" (Path.GetFullPath file)
        let outDir = Path.Combine(cfg.MovementDir, Path.GetFileNameWithoutExtension file)
        generateMovementForFile cfg.MovementCfgs file outDir


/// Records data for all */movement.json files in the given directory.
let recordMovements dir =
    let p = FsPickler.CreateJsonSerializer(indent=true)
    let bp = FsPickler.CreateBinarySerializer()
    
    for subDir in Directory.EnumerateDirectories dir do
        let movementFile = Path.Combine (subDir, "movement.dat")
        if File.Exists movementFile then
            printfn "%s" movementFile
            use tr = File.OpenRead movementFile
            let movement : Movement = bp.Deserialize tr
            use tr = File.OpenRead (Path.Combine (subDir, "curve.dat"))
            let curve : XY list = bp.Deserialize tr

            let driveCurve = toDriveCurve movement
            let tactileCurve = TactileCurve.record driveCurve
            use tw = File.OpenWrite (Path.Combine (subDir, "tactile.json"))
            p.Serialize (tw, tactileCurve)
            plotTactile (Path.Combine (subDir, "tactile.pdf")) curve tactileCurve

            let recMovement = syncTactileCurve tactileCurve movement
            use tw = File.OpenWrite (Path.Combine (subDir, "recorded.dat"))
            bp.Serialize (tw, recMovement)

            plotRecordedMovement (Path.Combine (subDir, "recorded.pdf")) curve recMovement None


let plotRecordedMovements dir =
    let bp = FsPickler.CreateBinarySerializer()
    
    for subDir in Directory.EnumerateDirectories dir do
        let recordedFile = Path.Combine (subDir, "recorded.dat")
        if File.Exists recordedFile then
            printfn "%s" recordedFile
            use tr = File.OpenRead recordedFile
            let recMovement : RecordedMovement = bp.Deserialize tr
            use tr = File.OpenRead (Path.Combine (subDir, "curve.dat"))
            let curve : XY list = bp.Deserialize tr

            plotRecordedMovement (Path.Combine (subDir, "recorded.pdf")) curve recMovement None

            
