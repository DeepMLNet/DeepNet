namespace MLPlots
open ArrayNDNS
open RProvider
open RProvider.graphics
open RProvider.grDevices

[<AutoOpen>]
module Utils =
    let i =  ignore

    /// Transforms an ArrayND<single> to a float list that can be used by RProvider
    let toFloatList (x:ArrayNDT<single>)= 
        match x with
        | :? ArrayNDCudaT<single> as ca -> ca  |> ArrayNDCuda.toHost |> ArrayNDHost.toList |> List.map (fun x -> float x)
        | :? ArrayNDHostT<single>  as ha-> ha |> ArrayNDHost.toList |> List.map (fun x -> float x)
        | _ -> failwith "Function not yet implemented for this subtype" 
    
    /// Saves a plot in directory dir with name name and size height x width
    let savePlot (height:int) (width:int) (dir:string) (name:string) (plot:unit-> unit) =
        let path = dir + @"/" + name 
        R.png(filename = path, height = height,width = width) |> i
        plot()
        R.dev_off ()

