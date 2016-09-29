namespace MLPlots

open System.IO

open ArrayNDNS
open RProvider
open RProvider.graphics
open RProvider.grDevices

[<AutoOpen>]
module Utils =
    let i =  ignore

    /// Transforms an ArrayND<single> to a float list that can be used by RProvider
    let toFloatList (x: ArrayNDT<single>) : float list = 
        x |> ArrayNDHost.fetch |> ArrayNDHost.convert |> ArrayNDHost.toList
    
    /// Saves a plot in directory dir with name name and size height x width
    let savePlot (height:int) (width:int) (dir:string) (name:string) (plot:unit-> unit) =
        let path = dir + @"/" + name 
        match Path.GetExtension path with
        | ".png" -> R.png (filename=path, height=height, width=width) |> i
        | ".pdf" -> R.pdf (path) |> i
        | ext -> failwithf "unsupported extension: %s" ext
        plot()
        R.dev_off () |> i

