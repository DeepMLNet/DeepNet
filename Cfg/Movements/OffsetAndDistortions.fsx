#if !CONFIG
#I "../../CurveFollow/bin/Debug"
#endif 

#r "CurveFollow.exe"

open Movement


let baseMovement = {
    Movement.Dt             = 0.01
    Movement.IndentorPos    = -43.4      // python had: -43.4        
    Movement.Accel          = 40.0 
    Movement.VelX           = 6.0 
    Movement.MaxVel         = 40.0
    Movement.MaxControlVel  = 15.0

    Movement.Mode           = Movement.FixedOffset 0.
}

let distortionMode = 
    Movement.Distortions {
        Movement.DistortionsPerSec = 0.9
        Movement.MaxOffset         = 3.0
        Movement.MinHold           = 0.6
        Movement.MaxHold           = 2.0
        Movement.NotAgainFor       = 20.0
    }         



let cfg = {
    CurveDir        = "../../Data/DeepBraille/Curves/test1"
    MovementDir     = "../../Data/DeepBraille/Movements/test1"
    MovementCfgs    = 
        [
            {baseMovement with Mode = distortionMode}
        ]
}
