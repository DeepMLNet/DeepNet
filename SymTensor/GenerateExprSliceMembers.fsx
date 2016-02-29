
open System
open System.IO

let maxDim = 3

let f = new StreamWriter ("ExprSliceMembers.txt")
let ws = "        "

let prn fmt =
    let write str = 
        f.WriteLine (ws + str)
    Printf.kprintf write fmt

type RangeSpecValueT =
    | RSVInt
    | RSVSpecial   // newaxis, fill
    | RSVSizeSpec
    | RSVExpr
    | RSVPlusElems
    | RSVObj

type RangeSpecT =
    | Item of RangeSpecValueT
    | Slice of RangeSpecValueT * RangeSpecValueT

let rangeSpecCombis = [
    Item RSVSizeSpec;
    Item RSVInt;
    Item RSVSpecial;
    Item RSVExpr;

    Slice (RSVObj, RSVObj);
    //Slice (SSVSizeSpec, SSVSizeSpec);
    //Slice (SSVSizeSpec, SSVInt);
    //Slice (SSVInt, SSVSizeSpec);
    //Slice (SSVInt, SSVInt);
    //Slice (SSVExpr, SSVPlusElems);
]


let typeAndUnionCase rs =                                          
    match rs with
    | RSVInt -> "int", "RSVInt"
    | RSVSpecial -> "SpecialAxisT", "RSVSpecial"
    | RSVSizeSpec -> "SizeSpecT", "RSVSizeSpec" 
    | RSVExpr -> "ExprT<int>", "RSVExpr"
    | RSVPlusElems -> "PlusElemsT", "RSVPlusElems"
    | RSVObj -> "#obj", "RSVObj"



prn "// ========================= ITEM / SLICE MEMBERS BEGIN ============================="

for dim in 1 .. maxDim do
    let ad = {0 .. dim-1}

    let rec generate rangeSpecs = 
        if List.length rangeSpecs < dim then
            for rs in rangeSpecCombis do
                generate (rs :: rangeSpecs)
        else

            let decls, args = 
                rangeSpecs 
                |> List.mapi (fun dim rs ->
                    match rs with
                    | Item is -> 
                        let t, uc = typeAndUnionCase is
                        sprintf "d%d: %s" dim t, sprintf "Item (%s d%d)" uc dim
                    | Slice (ss, fs) ->
                        let st, suc = typeAndUnionCase ss
                        let ft, fuc = typeAndUnionCase fs
                        sprintf "d%ds: %s option, d%df: %s option" dim st dim ft,
                        sprintf "RangeSpecT.ObjSlice d%ds d%df" dim dim
                    )
                |> List.unzip

            if rangeSpecs |> List.exists (function | Slice _ -> true | _ -> false) then
                prn "member this.GetSlice (%s) = this.BuildSlice [%s]" 
                    (String.concat ", " decls) (String.concat "; " args)
            else
                prn "member this.Item with get (%s) = this.BuildSlice [%s]"
                    (String.concat ", " decls) (String.concat "; " args)

    prn ""
    prn "// %d dimensions" dim
    generate []


prn "// ========================= ITEM / SLICE MEMBERS END ==============================="

f.Dispose ()


        
