namespace ModelPlots
open ArrayNDNS
open SymTensor
open RProvider
open RProvider.graphics
open RProvider.ggplot2
open RTools

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
        namedParams [   
            "x", box x;
             "y", box y;
             "col", box "red";
             "type", box "n"]
        |> R.plot |> i
        namedParams [   
            "x", box (x @ xrev);
             "y", box (ydown@ yup);
             "col", box "skyblue";
             "border" , box "NA"]
        |> R.polygon |>i  
        R.lines2 (x, y, "black")
        ()

