module XYTableSim

type XY = float * float

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

let optimalVelToPos tPos maxControlVel cfg state =
    let tPosX, tPosY = tPos
    let maxControlVelX, maxControlVelY = maxControlVel
    let {Accel=accelX, accelY} = cfg
    let {Pos=posX, posY} = state
    let vx = optimalVelAxis tPosX posX accelX maxControlVelX cfg.Dt
    let vy = optimalVelAxis tPosY posY accelY maxControlVelY cfg.Dt
    vx, vy
