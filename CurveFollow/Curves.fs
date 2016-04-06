module Curves


open Basics
open ArrayNDNS


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


module Movement = 

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
        Accel:          float
        Points:         MovementPoint list
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
            Accel       = cfg.Accel
            Points      = movement
        }
