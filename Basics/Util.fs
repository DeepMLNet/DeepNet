namespace Basics


open System


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

    /// combines sequence of string with given seperator but returns empty if sequence is empty
    let concatButIfEmpty empty sep items =
        if Seq.isEmpty items then empty
        else String.concat sep items



[<AutoOpen>]
module UtilTypes =

    [<Measure>]
    type bytes

    [<Measure>]
    type elements

    type Dictionary<'TKey, 'TValue> = System.Collections.Generic.Dictionary<'TKey, 'TValue>

    let conv<'T> value : 'T =
        Convert.ChangeType(box value, typeof<'T>) :?> 'T

    /// Default value for options. Returns b if a is None, else the value of a.
    let inline (|?) (a: 'a option) b = if a.IsSome then a.Value else b

module Util =

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

    /// path to application directory under AppData\Local
    let localAppData =  
        let lad = System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData)
        System.IO.Path.Combine (lad, "expr2")
    
    /// converts sequence of ints to sequence of strings
    let intToStrSeq items =
        Seq.map (sprintf "%d") items

    /// C++ data type for given type
    let cppType (typ: System.Type) = 
        match typ with
        | _ when typ = typeof<double>   -> "double"
        | _ when typ = typeof<single>   -> "float"
        | _ when typ = typeof<int>      -> "int"
        | _ when typ = typeof<byte>     -> "char"
        | _ -> failwithf "no C++ datatype for %A" typ


