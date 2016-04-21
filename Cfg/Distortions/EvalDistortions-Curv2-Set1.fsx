#if !CONFIG
#I "../../CurveFollow/bin/Debug"
#endif 

#r "Models.dll"
#r "CurveFollow.exe"

open ControllerEval
open Models


let baseCurve = {
    ControllerEval.Cfg.Dx             = 0.1
    ControllerEval.Cfg.VelX           = 6.0
    ControllerEval.Cfg.Mode           = NoDistortions
    ControllerEval.Cfg.IndentorPos    = -43.4      
    ControllerEval.Cfg.FollowCurveToX = 20.0      
}

let distortionMode = 
    ControllerEval.DistortionsAtXPos {
        MaxOffset           = 3.0
        Hold                = 15.0
        XPos                = [30.0; 80.0]
    }         


let cfg = {
    CurveDir        = Config.baseDir + "/Data/DeepBraille/Curves/curv2"
    DistortionDir   = Config.baseDir + "/Data/DeepBraille/Distortions/Curv2-Set1"
    DistortionCfgs  = 
        [
            baseCurve

            {baseCurve with Mode = distortionMode}
            {baseCurve with Mode = distortionMode}
            {baseCurve with Mode = distortionMode}
        ]
}
