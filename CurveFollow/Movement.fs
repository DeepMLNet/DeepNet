module Movement

open System
open System.IO

open Basics
open ArrayNDNS

open RProvider
open RProvider.graphics
open RProvider.grDevices

type XY = float * float


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
            if abs(tVel) > maxVel then float (sign tVel) * maxVel
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

    let private optimalVelAxis tPos pos accel maxVel = 
        let d = tPos - pos
        let accel = if abs d < 0.1 then accel / 4.0 else accel
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
        let vx = optimalVelAxis tPosX posX accelX maxControlVelX
        let vy = optimalVelAxis tPosY posY accelY maxControlVelY
        vx, vy



type DistortionCfg = {
    DistortionsPerSec:      float
    MaxOffset:              float
    MaxHold:                float
}

type private DistortionState = 
    | Inactive
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
            Distorted   = distState <> Inactive
        }
        let x, y = state.Pos
        match curve with
        | [] -> ()
        | (x1,y1) :: (x2,y2) :: _ when x1 <= x && x < x2 ->
            // interpolate curve points
            let fac = (x2 - x) / (x2 - x1)
            let cy = fac * y2 + (1. - fac) * y1

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
                | Inactive ->               
                    let prob = dc.DistortionsPerSec * cfg.Dt
                    if rnd.NextDouble() < prob then
                        let trgt = rnd.NextDouble() * dc.MaxOffset
                        let trgt = min trgt (baseY + 0.5)
                        let trgt = max trgt (baseY - 0.5)
                        yield! generate curve state (GotoPos trgt)
                    else 
                        yield movementPoint optVel optVel
                        yield! generate curve state Inactive
                | GotoPos trgt ->
                    if abs (y - trgt) < 0.05 then
                        let hu = x + rnd.NextDouble() * dc.MaxHold
                        yield! generate curve state (HoldUntil hu)
                    else
                        let _, cVelY = OptimalControl.toPos (0., trgt) (0., cfg.MaxControlVel) tblCfg state
                        let cVel = cfg.VelX, cVelY

                        yield movementPoint cVel optVel
                        yield! generate curve (XYTableSim.step cVel tblCfg state) distState
                | HoldUntil hu ->
                    if x >= hu then
                        yield! generate curve state Inactive
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
    let movement = generate curve state Inactive |> Seq.toList 

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
                let x = (1.0 - fac) * xa + fac * xb
                let y = (1.0 - fac) * ya + fac * yb
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
              yield [for idx = 0 to pos.Shape.[1] do
                         yield pos.[[0; idx; smpl]], pos.[[1; idx; smpl]]] }  
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

    R.plot (namedParams ["x",    box curveX
                         "y",    box curveY
                         "col",  box "black"
                         "xlim", box [0; 24]
                         "ylim", box [curveY.[0] - 0.6; curveY.[0] + 0.6]])    |> ignore

    R.abline(h=curveY.[0]) |> ignore
    R.lines (namedParams ["x",    box posX
                          "y",    box posY
                          "col",  box "blue"])   |> ignore

    R.dev_off() |> ignore

//
//    sg = multifig([4, 2])
//
//    plt.subplot(sg[0])
//    plt.ylim(curve[0, 1] - 0.6, curve[0, 1] + 0.6)
//    shade_regions(curve[:, 0], distorted)
//    plt.plot(curve[:, 0], curve[0, 1] * np.ones_like(curve[:, 0]), 'k')
//    plt.plot(curve[:, 0], curve[:, 1], 'b-', label="curve")
//    plt.plot(movement_pos[:, 0], movement_pos[:, 1], 'r.', label="movement")
//    plt.xlim(0, 24)
//
//    plt.xlabel("column")
//    plt.ylabel("row")
//    plt.title("position")
//
//    plt.subplot(sg[1])
//    plt.ylim(-2, 2)
//    shade_regions(curve[:, 0], distorted)
//    plt.plot(movement_control_vel[:, 0], movement_control_vel[:, 1], 'b.-', label='control')
//    plt.plot(movement_optimal_vel[:, 0], movement_optimal_vel[:, 1], 'b.-', label='optimal')
//    plt.legend()
//    plt.xlabel('column')
//    plt.ylabel('velocity')
//    plt.xlim(0, 24)
//    plt.title("velocity")
//
//    plt.tight_layout()
//    plt.savefig(path)
//    plt.close()


let generateMovementForFile cfgs path =
    let rnd = Random ()
    let baseDir = Path.Combine(Path.GetDirectoryName path, Path.GetFileNameWithoutExtension path)
    let curves = loadCurves path
    for curveIdx, curve in List.indexed curves do
        for cfgIdx, cfg in List.indexed cfgs do
            let dir = Path.Combine(baseDir, sprintf "Curve%d" curveIdx, sprintf "Cfg%d" cfgIdx)
            Directory.CreateDirectory dir |> ignore

            let movement = generate cfg rnd curve
            let driveCurve = toDriveCurve movement

            // plot movement
            plotMovement (Path.Combine (dir, "movement.pdf")) curve movement
            
let generateMovementForDir cfgs dir =
    for file in Directory.EnumerateFiles(Path.Combine(dir, "*.cur.npz")) do
        generateMovementForFile cfgs file


