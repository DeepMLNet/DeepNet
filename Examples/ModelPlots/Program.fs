// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
namespace ModelTests

module Program =
    [<EntryPoint>]
    let main argv = 
//        SymTensor.Debug.VisualizeUExpr <- true
//        ApplicationTests.testPlot ()
//        PlotTests.multiplotTest ()
        GPTests.EPTest ()
        PlotTests.GPTransferTest ()
        0 // return an integer exit code
