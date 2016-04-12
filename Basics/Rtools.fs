namespace Basics

open RProvider
open RProvider.graphics


module RCall =
    let empty : (string * obj) list = []
    let param (name: string) value args = 
        match value with
        | Some xval -> (name, box xval) :: args
        | None -> args
    let call func args =
        args |> namedParams |> func |> ignore


type R () =

    static member plot2 (?xlim, ?ylim, ?title, ?xlabel, ?ylabel) =
        RCall.empty
        |> RCall.param "xlim" xlim
        |> RCall.param "ylim" ylim
        |> RCall.param "main" title
        |> RCall.param "xlab" xlabel
        |> RCall.param "ylab" ylabel
        |> RCall.param "x" (Some (R.vector()))
        |> RCall.param "y" (Some (R.vector()))
        |> RCall.call R.plot

    static member lines2 (?x, ?y, ?color) =
        RCall.empty
        |> RCall.param "x" x
        |> RCall.param "y" y
        |> RCall.param "col" color
        |> RCall.call R.lines

    static member par2 (param, value) =
        RCall.empty
        |> RCall.param param (Some value)
        |> RCall.call R.par
