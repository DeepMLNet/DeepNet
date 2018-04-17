namespace MLPlots

open System.IO
open Tensor
open RProvider
open RProvider.graphics
open RProvider.grDevices
open RTools

[<AutoOpen>]
module Utils =

    let private ig =  ignore


    /// Transforms an ArrayND<single> to a float list that can be used by RProvider.
    let toFloatList (x: Tensor<single>) : float list = 
        x |> HostTensor.transfer |> Tensor.convert |> HostTensor.toList
    

    /// Saves a plot in directory dir with name name and size height x width.
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
    

    /// Plots several plots in one image.
    /// Takes list of tuple
    /// plot name:      string
    /// plot function:  (unit -> unit)
    let plotgrid (plots:list<string*(unit-> unit)>) = 
        R.lock (fun () ->
            let nPlots = List.length plots |> float
            let shape = 
                let side1 = ceil(sqrt nPlots)
                let side2 =  side1 //ceil(nPlots/side1)
                let side1,side2 = (int side1), (int side2)
                [side2;side1]
            R.par2 ("mfrow", shape)
            R.par2 ("mar",box [1.0;1.0;1.0;1.0])
            |> R.par |> ig
            plots |> List.map (fun (name, plot) -> 
                        plot()
                        namedParams[
                            "main", name]
                        |> R.title |>ig) |> ig
        )



