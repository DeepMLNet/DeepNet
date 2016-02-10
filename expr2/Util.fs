module Util

[<Measure>]
type bytes

[<Measure>]
type elements

type Dictionary<'a, 'b> = System.Collections.Generic.Dictionary<'a, 'b>

module List =
    /// sets element with index elem to given value
    let rec set elem value lst =
        match lst, elem with
            | l::ls, 0 -> value::ls
            | l::ls, _ -> l::(set (elem-1) value ls)
            | [], _ -> invalidArg "elem" "element index out of bounds"

    /// removes element with index elem 
    let without elem lst =
        List.concat [List.take elem lst; List.skip (elem+1) lst] 

    /// removes all elements with the given value
    let withoutValue value lst =
        lst |> List.filter ((<>) value)

    /// removes the first element with the given value
    let rec removeValueOnce value lst =
        match lst with
        | v::vs when v = value -> vs
        | v::vs -> v :: removeValueOnce value vs
        | [] -> []

    /// insert the specified value at index elem
    let insert elem value lst =
        List.concat [List.take elem lst; [value]; List.skip elem lst]

module Map = 
    /// adds all items from q to p
    let join (p:Map<'a,'b>) (q:Map<'a,'b>) = 
        Map(Seq.concat [ (Map.toSeq p) ; (Map.toSeq q) ])    

module String =
    /// concatenates the given items with sep inbetween
    let combineWith sep items =    
        let rec combine items = 
            match items with
            | [item] -> item
            | item::rest -> item + sep + combine rest
            | [] -> ""
        items |> Seq.toList |> combine

/// iterates function f n times
let rec iterate f n x =
    match n with
    | 0 -> x
    | n when n > 0 -> iterate f (n-1) (f x)
    | _ -> failwithf "cannot execute negative iterations %d" n

/// directory of our assembly
let assemblyDirectory = 
    // http://stackoverflow.com/questions/52797/how-do-i-get-the-path-of-the-assembly-the-code-is-in
    let codeBase = System.Reflection.Assembly.GetExecutingAssembly().CodeBase
    let uri = new System.UriBuilder(codeBase)
    let path = System.Uri.UnescapeDataString(uri.Path)
    System.IO.Path.GetDirectoryName(path)

/// converts sequence of ints to sequence of strings
let intToStrSeq items =
    Seq.map (sprintf "%d") items
