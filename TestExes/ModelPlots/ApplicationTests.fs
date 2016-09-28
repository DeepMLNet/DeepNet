namespace ModelPlots
open ArrayNDNS
open SymTensor
open RProvider
open RProvider
open RProvider.graphics
open RProvider.ggplot2

module ApplicationTests =
    let i = ignore
    let testPlot () =
        let x = [-5.0..0.1..5.0]
        let y = x|> List.map (fun x -> 0.5* x**2.0)
        let yup = y |> List.map (fun x -> x+1.5)
        let ydown = y |> List.map (fun x -> x-1.5) |> List.rev
        let xrev = x |> List.rev
        printfn "x = %A" x
        printfn "y = %A" y
//        let data = R.data_frame(x,y)
        R.plot(x, y) |> i
        namedParams [   
            "x", box x; 
            "y", box y; 
            "type", box "l"; 
            "col", box "red";
            "ylim", box [-2; 15]]
        |> R.plot |> i
        namedParams [   
            "x", box (x @ xrev);
            "y", box (ydown@ yup);
            "col", box "skyblue"]
        |> R.polygon |> i
        ()

