namespace SymTensor


/// Utility functions
module internal Util =

    /// matches integral values (e.g. 2, 2.0 or 2.0f, etc.)
    let (|Integral|_|) (x: 'T) =
        match typeof<'T> with
        | t when t = typeof<int> ->
            Some (x |> box |> unbox<int>)
        | t when t = typeof<byte> ->
            Some (x |> box |> unbox<byte> |> int)
        | t when t = typeof<float> ->
            let f = x |> box |> unbox<float>
            if abs (f % 1.0) < System.Double.Epsilon then
                Some (f |> round |> int)
            else None
        | t when t = typeof<single> ->
            let f = x |> box |> unbox<single>
            if abs (f % 1.0f) < System.Single.Epsilon then
                Some (f |> round |> int)
            else None
        | _ -> None

    /// part of a limited string length format specifier
    type LimitedStringPart =
        /// a delimiter that is always inserted
        | Delim of string
        /// a formatter that is replaced by "..." if string becomes too long
        | Formatter of (int -> string)

    /// builds a string of approximate maximum length from the given format specifier parts
    let limitedToString maxLength parts =
        ("", parts)
        ||> List.fold (fun s p ->
            match p with
            | Delim d -> s + d
            | Formatter fmtFn -> 
                match maxLength - s.Length with
                | remLength when remLength > 0 -> s + fmtFn remLength
                | _ -> s + "..."
            )


/// Utility operators        
[<AutoOpen>]
module internal UtilOperators = 

    /// Default value for options. Returns b if a is None, else the value of a.
    let inline (|?) (a: 'a option) b = 
        match a with
        | Some v -> v
        | None -> b
                      


/// Extensions to basic collection types.
[<AutoOpen>]
module internal Extensions =

    type System.Collections.Generic.HashSet<'T> with
        member this.LockedContains key =
            lock this (fun () -> this.Contains key)

        member this.LockedAdd key =
            lock this (fun () -> this.Add key)

    type System.Collections.Generic.IReadOnlyDictionary<'TKey, 'TValue> with
        member this.TryFindReadOnly key =
            let value = ref (Unchecked.defaultof<'TValue>)
            if this.TryGetValue (key, value) then Some !value
            else None

    type System.Collections.Generic.Queue<'T> with
        member this.TryPeek =
            if this.Count > 0 then Some (this.Peek())
            else None

