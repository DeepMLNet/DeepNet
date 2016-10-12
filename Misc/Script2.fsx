let orderedStride (shape: int list) (order: int list) =
    let rec build cumElems order =
        match order with
        | o :: os -> cumElems :: build (cumElems * shape.[o]) os
        | [] -> []
    build 1 order |> List.permute (fun i -> order.[i])


orderedStride [6; 3; 4] [2; 1; 0]
orderedStride [6; 3; 4] [0; 1; 2]
orderedStride [6; 3; 4] [0; 2; 1]
orderedStride [6; 3; 4] [2; 0; 1]

