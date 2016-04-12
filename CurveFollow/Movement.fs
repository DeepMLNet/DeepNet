module Movement

open System
open System.IO
open RProvider
open RProvider.graphics
open RProvider.grDevices
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
    let col_offset = 3.0        // chars
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



module XYTableSim =

    type State = {
        Time:  float
        Pos:   XY
        Vel:   XY     
    }

    type Cfg = {
        Accel:  XY
        MaxVel: XY
        Dt:     float
    }

    let private stepAxis tVel pos vel accel maxVel dt =
        let tVel = 
            if abs tVel > maxVel then float (sign tVel) * maxVel
            else tVel
        let dVel = tVel - vel
        let vel = 
            if abs dVel < accel * dt then tVel
            else vel + accel * (sign dVel |> float) * dt
        let pos = pos + vel * dt
        pos, vel

    let step tVel cfg state =
        let tVelX, tVelY = tVel
        let {Accel=accelX, accelY; MaxVel=maxVelX, maxVelY; Dt=dt} = cfg
        let {Time=t; Pos=posX, posY; Vel=velX, velY} = state
        let t = t + dt
        let posX, velX = stepAxis tVelX posX velX accelX maxVelX dt
        let posY, velY = stepAxis tVelY posY velY accelY maxVelY dt
        {Time=t; Pos=posX, posY; Vel=velX, velY}


module OptimalControl = 
    open XYTableSim

    let private optimalVelAxis tPos pos accel maxVel dt = 
        let d = tPos - pos
        if abs d < 0.2 then
            d / (dt * 5.)
        else
            let accel = if abs d < 0.7 then accel / 4.0 else accel
            let stopVel = sqrt (2.0 * accel * abs d) * float (sign d)
            let vel = 
                if abs stopVel > maxVel then float (sign d) * maxVel
                else stopVel
            vel        

    let toPos tPos maxControlVel cfg state =
        let tPosX, tPosY = tPos
        let maxControlVelX, maxControlVelY = maxControlVel
        let {Accel=accelX, accelY} = cfg
        let {Pos=posX, posY} = state
        let vx = optimalVelAxis tPosX posX accelX maxControlVelX cfg.Dt
        let vy = optimalVelAxis tPosY posY accelY maxControlVelY cfg.Dt
        vx, vy



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
    OptimalVel:     XY
    Distorted:      bool
}

type Movement = {
    StartPos:       XY
    IndentorPos:    float
    Accel:          float
    Points:         MovementPoint list
}

type RecordedMovementPoint = {
    Time:           float
    SimPos:         XY
    DrivenPos:      XY
    ControlVel:     XY
    OptimalVel:     XY
    Distorted:      bool
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

    let rec generate curve (state: XYTableSim.State) distState = seq {
        let movementPoint cVel optVel = {
            Time        = state.Time
            Pos         = state.Pos
            ControlVel  = cVel
            OptimalVel  = optVel
            Distorted   = match distState with InactiveUntil _ -> true | _ -> false
        }
        let x, y = state.Pos
        match curve with
        | [] -> ()
        | (x1,y1) :: (x2,y2) :: _ when x1 <= x && x < x2 ->
            // interpolate curve points
            let fac = (x2 - x) / (x2 - x1)
            let cy = fac * y1 + (1. - fac) * y2

            // optimal velocity to track curve
            let _, optVelY = OptimalControl.toPos (0., cy) (0., cfg.MaxControlVel) tblCfg state
            let optVel = cfg.VelX, optVelY

            match cfg.Mode with
            | FixedOffset ofst ->         
                let _, cVelY = OptimalControl.toPos (0., cy + ofst) (0., cfg.MaxControlVel) tblCfg state
                let cVel = cfg.VelX, cVelY

                yield movementPoint cVel optVel
                yield! generate curve (XYTableSim.step cVel tblCfg state) distState
            | Distortions dc ->
                match distState with
                | InactiveUntil iu ->               
                    let prob = dc.DistortionsPerSec * cfg.Dt
                    if x >= iu && rnd.NextDouble() < prob then
                        let trgt = cy + (2.0 * rnd.NextDouble() - 1.0) * dc.MaxOffset
                        let trgt = min trgt (baseY + 0.5)
                        let trgt = max trgt (baseY - 0.5)
                        yield! generate curve state (GotoPos trgt)
                    else 
                        yield movementPoint optVel optVel
                        yield! generate curve (XYTableSim.step optVel tblCfg state) distState
                | GotoPos trgt ->
                    if abs (y - trgt) < 0.05 then
                        let hu = x + dc.MinHold + rnd.NextDouble() * (dc.MaxHold - dc.MinHold)
                        yield! generate curve state (HoldUntil hu)
                    else
                        let _, cVelY = OptimalControl.toPos (0., trgt) (0., cfg.MaxControlVel) tblCfg state
                        let cVel = cfg.VelX, cVelY

                        yield movementPoint cVel optVel
                        yield! generate curve (XYTableSim.step cVel tblCfg state) distState
                | HoldUntil hu ->
                    if x >= hu then
                        yield! generate curve state (InactiveUntil (x + dc.NotAgainFor))
                    else
                        let cVel = cfg.VelX, 0.
                        yield movementPoint cVel optVel
                        yield! generate curve (XYTableSim.step cVel tblCfg state) distState

        | (x1,_) :: _ when x < x1 ->
            // x position is left of curve start
            let vel = cfg.VelX, 0.
            yield movementPoint vel vel
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
        Points      = movement
    }


let toDriveCurve (movement: Movement) = 
    {
        TactileCurve.IndentorPos = movement.IndentorPos
        TactileCurve.StartPos    = movement.StartPos
        TactileCurve.Accel       = movement.Accel
        TactileCurve.Points      = [ for mp in movement.Points -> 
                                     {
                                        TactileCurve.Time = mp.Time
                                        TactileCurve.Vel  = mp.ControlVel
                                     } ]            
    }


let syncTactileCurve (tc: TactileCurve.TactileCurve) (m: Movement) =
    let rec syncPoints (tcPoints: TactileCurve.TactilePoint list) (mPoints: MovementPoint list) = seq {
        match tcPoints, mPoints with
        | [], _ -> ()
        | _, [] -> ()
        | ({Time=tct} as t)::tcRest, ({Time=mt} as m)::({Time=mtNext} as mNext)::_ when mt <= tct && tct < mtNext ->
            let fac = float (mtNext - tct) / float (mtNext - mt)
            let interp a b = 
                let xa, ya = a
                let xb, yb = b
                let x = fac * xa + (1.0 - fac) * xb
                let y = fac * ya + (1.0 - fac) * yb
                x, y
            yield {
                Time       = tct
                SimPos     = interp m.Pos mNext.Pos
                DrivenPos  = t.Pos
                ControlVel = interp m.ControlVel mNext.ControlVel
                OptimalVel = interp m.OptimalVel mNext.OptimalVel
                Distorted  = m.Distorted
            }
            yield! syncPoints tcRest mPoints
        | {Time=tct}::tcRest, {Time=mt}::_ when tct < mt ->
            yield! syncPoints tcRest mPoints
        | _, _::mRest ->
            yield! syncPoints tcPoints mRest
    }
    
    {
        IndentorPos = tc.IndentorPos
        Accel       = tc.Accel
        Points      = syncPoints tc.Points m.Points |> List.ofSeq
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
    let optimalVelX, optimalVelY = toArray (fun (p: MovementPoint) -> p.OptimalVel) movement.Points
    let distorted = movement.Points |> List.map (fun p -> p.Distorted) |> Array.ofList

    R.pdf (path) |> ignore
    R.par2 ("oma", [0; 0; 0; 0])
    R.par2 ("mar", [3.2; 2.6; 1.0; 0.5])
    R.par2 ("mgp", [1.7; 0.7; 0.0])
    R.par2 ("mfrow", [2; 1])

    R.plot2 ([40; 190], [curveY.[0] - 6.; curveY.[0] + 6.], "position", "x", "y")
    R.abline(h=curveY.[0]) |> ignore
    R.lines2 (curveX, curveY, "black")
    R.lines2 (posX, posY, "blue")
    R.legend (155., curveY.[0] + 6., ["curve"; "movement"], col=["black"; "blue"], lty=[1;1]) |> ignore

    R.plot2 ([40; 190], [-20; 20], "velocity", "x", "y velocity")
    R.abline(h=0) |> ignore
    R.lines2 (posX, controlVelY, "blue")
    R.lines2 (posX, optimalVelY, "red")
    R.legend (165., 20, ["control"; "optimal"], col=["blue"; "red"], lty=[1;1]) |> ignore

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

            let p = FsPickler.CreateJsonSerializer(indent=true, omitHeader=true)
            use tw = File.OpenWrite(Path.Combine (dir, "movement.json"))
            p.Serialize(tw, movement)
            
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
    let p = FsPickler.CreateJsonSerializer(indent=true, omitHeader=true)
    
    for subDir in Directory.EnumerateDirectories dir do
        let movementFile = Path.Combine (subDir, "movement.json")
        if File.Exists movementFile then
            printfn "%s" movementFile
            use tr = File.OpenRead movementFile
            let movement : Movement = p.Deserialize tr
            let driveCurve = toDriveCurve movement

            let tactileCurve = TactileCurve.record driveCurve
            use tw = File.OpenWrite (Path.Combine (subDir, "tactile.json"))
            p.Serialize (tw, tactileCurve)

            let recMovement = syncTactileCurve tactileCurve movement
            use tw = File.OpenWrite (Path.Combine (subDir, "recorded.json"))
            p.Serialize (tw, recMovement)

