module Util

module List =
    let rec set elem value lst =
        match lst, elem with
            | l::ls, 0 -> value::ls
            | l::ls, _ -> l::(set (elem-1) value ls)
            | [], _ -> invalidArg "elem" "element index out of bounds"

    let without elem lst =
        List.concat [List.take elem lst; List.skip (elem+1) lst] 

    let insert elem value lst =
        List.concat [List.take elem lst; [value]; List.skip elem lst]

let rec iterate f n x =
    match n with
    | 0 -> x
    | n when n > 0 -> iterate f (n-1) (f x)
    | _ -> failwithf "cannot execute negative iterations %d" n

