namespace RTools

open Tensor.Utils
open MathNet.Numerics
open RProvider
open RProvider.graphics
open RProvider.grDevices


module RCall =
    let empty : (string * obj) list = []
    let param (name: string) value args = 
        match value with
        | Some xval -> (name, box xval) :: args
        | None -> args
    let call func args =
        args |> namedParams |> func |> ignore


type RColorMap =
    | Rainbow
    | Heat
    | Terrain
    | Topo
    | Cm
    | Gray


module RLock =
    let rLock = obj ()


type R () =  

    static member lock fn =
        lock RLock.rLock fn

    static member plot2 (?xlim, ?ylim, ?title, ?xlabel, ?ylabel) =
        RCall.empty
        |> RCall.param "xlim" xlim
        |> RCall.param "ylim" ylim
        |> RCall.param "xaxs" (Some "i")
        |> RCall.param "yaxs" (Some "i")
        |> RCall.param "main" title
        |> RCall.param "xlab" xlabel
        |> RCall.param "ylab" ylabel
        |> RCall.param "x" (Some (R.vector()))
        |> RCall.param "y" (Some (R.vector()))
        |> RCall.call R.plot

    static member plot3 (?xRng, ?yRng, ?title, ?xlabel, ?ylabel) =
        let xRng = defaultArg xRng (-6., 6.)
        let yRng = defaultArg yRng (-3., 3.)
        let title = defaultArg title ""
        let xlabel = defaultArg xlabel ""
        let ylabel = defaultArg ylabel ""
        let xlim = [|fst xRng; snd xRng|]
        let ylim = [|fst yRng; snd yRng|]
        R.plot2(xlim, ylim, title, xlabel, ylabel)

    static member lines2 (?x, ?y, ?color) =
        RCall.empty
        |> RCall.param "x" x
        |> RCall.param "y" y
        |> RCall.param "col" color
        |> RCall.call R.lines

    static member points2 (?x, ?y, ?pch, ?color, ?bg) =
        RCall.empty
        |> RCall.param "x" x
        |> RCall.param "y" y
        |> RCall.param "pch" pch
        |> RCall.param "col" color
        |> RCall.param "bg" bg
        |> RCall.call R.points

    static member points3 (?x: float[], ?y: float[], ?pch: int, ?color: string, ?bg: string) =
        R.points2(?x=x, ?y=y, ?pch=pch, ?color=color, ?bg=bg)

    static member fillBetween (x: float[], y1: float[], y2: float[], ?color) =
        if x.Length <> y1.Length || x.Length <> y2.Length then
            failwith "x, y1, y2 must be of same length"
        R.polygon (x=R.c(x, R.rev(x)), y=R.c(y2, R.rev(y1)), ?col=color, border="NA") |> ignore

    static member par2 (param: string, value: 'f) =
        RCall.empty
        |> RCall.param param (Some value)
        |> RCall.call R.par

    static member image2 (image, ?lim, ?xlim, ?ylim, ?colormap, ?title, ?xlabel, ?ylabel) =
        let nc = 512
        let cm =
            match colormap with
            | Some Rainbow -> Some (R.rainbow(nc))
            | Some Heat    -> Some (R.heat_colors(nc))
            | Some Terrain -> Some (R.terrain_colors(nc))
            | Some Topo    -> Some (R.topo_colors(nc))
            | Some Cm      -> Some (R.cm_colors(nc))
            | Some Gray    -> Some (R.gray_colors(nc, start=0.0, ``end``=1.0))
            | None         -> None

        let lim = lim |> Option.map (fun (a, b) -> [a; b])

        let xlim =
            match xlim with
            | Some xlim -> xlim
            | None -> 0., float (Array2D.length2 image)
        let ylim =
            match ylim with
            | Some ylim -> ylim
            | None -> 0., float (Array2D.length1 image)
        let x = Generate.LinearSpaced (image.GetLength 1, fst xlim, snd xlim)
        let y = Generate.LinearSpaced (image.GetLength 0, fst ylim, snd ylim)
        
        RCall.empty
        |> RCall.param "x" (Some x)
        |> RCall.param "y" (Some y)
        |> RCall.param "z" (Some (Array2D.transpose image))
        |> RCall.param "zlim" lim
        |> RCall.param "col" cm
        |> RCall.param "main" title
        |> RCall.param "xlab" xlabel
        |> RCall.param "ylab" ylabel
        |> RCall.param "useRaster" (Some true)
        |> RCall.call R.image


    static member pdfPage (filename: string, ?nPlots) =
        let nPlots = defaultArg nPlots 1
        R.pdf (filename) |> ignore 
        R.par2 ("oma", [0; 0; 0; 0])
        R.par2 ("mar", [3.2; 2.6; 1.0; 0.5])
        R.par2 ("mgp", [1.7; 0.7; 0.0])
        R.par2 ("mfrow", [nPlots; 1])

    static member multiplot (nPlots) =
        R.par2 ("oma", [0; 0; 0; 0])
        R.par2 ("mar", [3.2; 2.6; 1.0; 0.5])
        R.par2 ("mgp", [1.7; 0.7; 0.0])
        R.par2 ("mfrow", [nPlots; 1])
 
 