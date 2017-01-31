namespace SymTensor.Compiler

open Basics
open ArrayNDNS
open SymTensor
open UExprTypes


[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module UExprVisualizer =

    let addManikins (uExpr: UExprT) 
                    (trgtManikins: Map<string, ArrayNDManikinT>) 
                    (srcManikins: ArrayNDManikinT list) =
        if Debug.VisualizeUExpr then           
            let strsForManikin id (manikin: ArrayNDManikinT) =
                let mStr = sprintf "%s: %A %A" id manikin.TypeName manikin.Shape
                let sStr = sprintf "%A" manikin.Storage
                id, mStr, sStr
            let trgtStrs = 
                trgtManikins 
                |> Map.toList
                |> List.map (fun (id, manikin) -> strsForManikin id manikin)
            let srcStrs = 
                srcManikins 
                |> List.indexed 
                |> List.map (fun (src, manikin) -> strsForManikin (sprintf "Src%d" src) manikin)
            UExprVisualizer.getActive().AddManikins uExpr (srcStrs @ trgtStrs)
            


