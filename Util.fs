module Util

module List =
    let rec set lst elem value =
        match lst, elem with
            | l::ls, 0 -> value::ls
            | l::ls, _ -> l::(set ls (elem-1) value)
            | [], _ -> invalidArg "elem" "element index out of bounds"

    let without elem lst =
        List.concat [List.take elem lst; List.skip (elem+1) lst] 

