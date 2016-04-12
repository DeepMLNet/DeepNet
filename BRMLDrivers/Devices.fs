namespace BRML.Drivers


module Config = 
    
    let XYTable : XYTableCfgT = {
            PortName       = "COM5";
            PortBaud       = 115200;
            X              = { StepperConfig = {Id=1; AnglePerStep=1.8; StepMode=8; StartVel=1000.;}
                               DegPerMM      = 360. / 1.25;
                               Home          = Stepper.Right;
                               MaxPos        = 147.;}
            Y              = { StepperConfig = {Id=2; AnglePerStep=1.8; StepMode=8; StartVel=1000.;}
                               DegPerMM      = 360. / 1.25;
                               Home          = Stepper.Left;
                               MaxPos        = 140.;}
            DefaultVel     = 30.;
            DefaultAccel   = 30.;
            HomeVel        = 10.;    
    }

    let Linmot: LinmotCfgT = {
            PortName       = "COM6";
            PortBaud       = 57600;
            Id             = 0x11;    
            DefaultVel     = 50.0;
            DefaultAccel   = 200.0;    
    }

    let Biotac: BioTacCfgT = {
            Cheetah        = uint32 1364033083
            Index          = 0
    }


module Devices =
    let XYTable = new XYTableT(Config.XYTable)
    let Linmot = new LinmotT(Config.Linmot)
    let Biotac = new BiotacT(Config.Biotac)

    let LinmotUpPos = -10.

    let init () =
        async {
            do! Linmot.Home () 
            do! Linmot.DriveTo LinmotUpPos
            do! XYTable.Home ()
        }
        |> Async.RunSynchronously
