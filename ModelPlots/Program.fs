// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
namespace ModelPlots
open ArrayNDNS
open SymTensor
open RProvider
open RProvider.ggplot2
module Program =
    [<EntryPoint>]
    let main argv = 
        ApplicationTests.testPlot ()
        printfn "%A" argv
        0 // return an integer exit code
