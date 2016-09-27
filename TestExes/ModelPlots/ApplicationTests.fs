namespace ModelPlots
open ArrayNDNS
open SymTensor
open RProvider
open RProvider.ggplot2
open RProvider.graphics

module ApplicationTests =
    let i = ignore
    let testPlot () =
        let x = [0.0..0.1..5.0]
        let y = x|> List.map (fun x -> 0.5* x**2.0 - 1.5 )
        printfn "x = %A" x
        printfn "y = %A" y
        let data = R.data_frame(x,y)
//        R.plot(x, y) |> i
//        namedParams [   
//            "x", box x; 
//            "type", box "l"; 
//            "col", box "blue";
//            "ylim", box [0; 25] ]
//        |> R.plot |> i
        namedParams [
            "data", box data
            "mapping", box (x,y);
            "colour", box "red";
            "size", box "3"]
        |> R.ggplot |>R.print |>i
        ()

