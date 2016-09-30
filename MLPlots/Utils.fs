namespace MLPlots

open System.IO

open ArrayNDNS
open RProvider
open RProvider.graphics
open RProvider.grDevices
open RTools

[<AutoOpen>]
module Utils =
    let ig =  ignore

    /// Transforms an ArrayND<single> to a float list that can be used by RProvider
    let toFloatList (x: ArrayNDT<single>) : float list = 
        x |> ArrayNDHost.fetch |> ArrayNDHost.convert |> ArrayNDHost.toList
    
    /// Saves a plot in directory dir with name name and size height x width
    let savePlot (height:int) (width:int) (dir:string) (name:string) (plot:unit-> unit) =
        let path = dir + @"/" + name 
        R.lock (fun () ->
            match Path.GetExtension path with
            | ".png" -> R.png (filename=path, height=height, width=width) |> ig
            | ".pdf" -> R.pdf (path) |> ig
            | ext -> failwithf "unsupported extension: %s" ext
            plot()
            R.dev_off () |> ig
        )

    let plotgrid perRow (plots:list<string*(unit-> unit)>) = 
        R.lock (fun () ->
            let nPlots = List.length plots
            let shape = 
                if nPlots <perRow then
                    [nPlots;1]
                else if nPlots % perRow = 0 then
                    [perRow;nPlots/perRow ]
                else
                    [perRow;nPlots/perRow + 1]
            printfn "Plot shape = %A" shape
            R.par2 ("mfrow", shape)
            R.par2 ("mar",box [1.0;1.0;1.0;1.0])
            |> R.par |> ig
            plots |> List.map (fun (name, plot) -> 
                        plot()
                        namedParams[
                            "main", name]
                        |> R.title |>ig) |> ig
        )



