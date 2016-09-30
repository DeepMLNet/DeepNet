// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
namespace ModelPlots
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open RProvider
open RProvider.ggplot2
module Program =
    [<EntryPoint>]
    let main argv = 
//        ApplicationTests.testPlot ()
        PlotTests.multiplotTest () 
        PlotTests.GPTransferTest ()
        0 // return an integer exit code
