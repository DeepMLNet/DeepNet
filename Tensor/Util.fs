namespace Basics

open System
open System.Reflection
open System.IO
open System.Runtime.InteropServices
open System.Collections.Concurrent
open FSharp.Reflection



module Seq = 

    /// every n-th element of the given sequence
    let everyNth n (input:seq<_>) = 
      seq { use en = input.GetEnumerator()
            // Call MoveNext at most 'n' times (or return false earlier)
            let rec nextN n = 
              if n = 0 then true
              else en.MoveNext() && (nextN (n - 1)) 
            // While we can move n elements forward...
            while nextN n do
              // Retrun each nth element
              yield en.Current }

    /// shuffles a finite sequence using the given seed
    let shuffle seed (input:seq<_>) =
        let rand = System.Random seed

        let swap (a: _[]) x y =
            let tmp = a.[x]
            a.[x] <- a.[y]
            a.[y] <- tmp

        let a = Array.ofSeq input
        Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a        
        a |> Seq.ofArray


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

    /// transposes a list list
    let rec transpose = function
        | (_::_)::_ as m -> List.map List.head m :: transpose (List.map List.tail m)
        | _ -> []


module Map = 
    /// adds all items from q to p
    let join (p:Map<'a,'b>) (q:Map<'a,'b>) = 
        Map(Seq.concat [(Map.toSeq p); (Map.toSeq q)])    

    /// Creates a map from a System.Collection.Generic.Dictionary<_,_>.
    let ofDictionary dictionary = 
        (dictionary :> seq<_>)
        |> Seq.map (|KeyValue|)
        |> Map.ofSeq

module String =

    /// combines sequence of string with given seperator but returns empty if sequence is empty
    let concatButIfEmpty empty sep items =
        if Seq.isEmpty items then empty
        else String.concat sep items


module Array2D =

    /// returns a transposed copy of the matrix
    let transpose m = 
        Array2D.init (Array2D.length2 m) (Array2D.length1 m) (fun y x -> m.[x, y])




[<AutoOpen>]
module UtilTypes =

    [<Measure>]
    type bytes

    [<Measure>]
    type elements

    type System.Collections.Generic.HashSet<'T> with
        member this.LockedContains key =
            lock this (fun () -> this.Contains key)

        member this.LockedAdd key =
            lock this (fun () -> this.Add key)

    type System.Collections.Generic.Dictionary<'TKey, 'TValue> with
        member this.TryFind key =
            let value = ref (Unchecked.defaultof<'TValue>)
            if this.TryGetValue (key, value) then Some !value
            else None

        member this.LockedTryFind key =
            lock this (fun () -> this.TryFind key)

        member this.GetOrDefault key dflt =
            match this.TryFind key with
            | Some v -> v
            | None -> dflt

        member this.LockedAdd (key, value) =
            lock this (fun () -> this.Add (key, value))

        member this.LockedSet (key, value) =
            lock this (fun () -> this.[key] <- value)

    type System.Collections.Concurrent.ConcurrentDictionary<'TKey, 'TValue> with
        member this.TryFind key =
            let value = ref (Unchecked.defaultof<'TValue>)
            if this.TryGetValue (key, value) then Some !value
            else None

        member this.GetOrDefault key dflt =
            match this.TryFind key with
            | Some v -> v
            | None -> dflt

    type System.Collections.Generic.Queue<'T> with
        member this.TryPeek =
            if this.Count > 0 then Some (this.Peek())
            else None

    type Dictionary<'TKey, 'TValue> = System.Collections.Generic.Dictionary<'TKey, 'TValue>
    type HashSet<'T> = System.Collections.Generic.HashSet<'T>
    type Queue<'T> = System.Collections.Generic.Queue<'T>
    type ConcurrentDictionary<'TKey, 'TValue> = System.Collections.Concurrent.ConcurrentDictionary<'TKey, 'TValue>

    /// convert given value to specified type and return as obj
    let convTo (typ: System.Type) value =
        Convert.ChangeType(box value, typ)

    /// convert given value to type 'T
    let conv<'T> value : 'T =
        Convert.ChangeType(box value, typeof<'T>) :?> 'T

    /// Default value for options. Returns b if a is None, else the value of a.
    let inline (|?) (a: 'a option) b = 
        match a with
        | Some v -> v
        | None -> b

    let allBindingFlags = 
        BindingFlags.Public ||| BindingFlags.NonPublic ||| 
        BindingFlags.Static ||| BindingFlags.Instance

    type private GenericMethodDescT = {
        ContainingType:     string
        MethodName:         string
        GenericTypeArgs:    string list
    }

    let private genericMethodCache = ConcurrentDictionary<GenericMethodDescT, MethodInfo> ()

    /// Calls the specified method on the type 'U with the specified generic type arguments
    /// and the specified arguments in tupled form. Return value is of type 'R.
    let callGenericInst<'U, 'R> (instance: obj) (methodName: string) (genericTypeArgs: System.Type list) args =
        let gmd = {
            ContainingType  = typeof<'U>.AssemblyQualifiedName
            MethodName      = methodName
            GenericTypeArgs = genericTypeArgs |> List.map (fun t -> t.AssemblyQualifiedName)
        }

        let m =
            match genericMethodCache.TryFind gmd with
            | Some m -> m
            | None ->
                let gm = typeof<'U>.GetMethod (methodName, allBindingFlags)
                if gm = null then
                    failwithf "cannot find method %s on type %A" methodName typeof<'U>
                let m = gm.MakeGenericMethod (List.toArray genericTypeArgs)
                genericMethodCache.[gmd] <- m
                m

        let args = FSharpValue.GetTupleFields args
        m.Invoke (instance, args) :?> 'R       

    /// Calls the specified static method on the type 'U with the specified generic type arguments
    /// and the specified arguments in tupled form. Return value is of type 'R.
    let callGeneric<'U, 'R> (methodName: string) (genericTypeArgs: System.Type list) args =
        callGenericInst<'U, 'R> null methodName genericTypeArgs args


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
        System.IO.Path.Combine (lad, "DeepNet")
    
    /// converts sequence of ints to sequence of strings
    let intToStrSeq items =
        Seq.map (sprintf "%d") items

    /// C++ data type for given type
    let cppType (typ: System.Type) = 
        match typ with
        | _ when typ = typeof<double>   -> "double"
        | _ when typ = typeof<single>   -> "float"
        | _ when typ = typeof<int32>    -> "int"
        | _ when typ = typeof<uint32>   -> "unsigned int"
        | _ when typ = typeof<byte>     -> "unsigned char"
        | _ when typ = typeof<bool>     -> "bool"
        | _ -> failwithf "no C++ datatype for %A" typ

    /// Returns the contents of a blittable structure as a byte array.
    let structToBytes (s: 'S when 'S: struct) =
        let size = Marshal.SizeOf(typeof<'S>)
        let byteAry : byte[] = Array.zeroCreate size

        let tmpPtr = Marshal.AllocHGlobal(size)
        Marshal.StructureToPtr(s, tmpPtr, false)
        Marshal.Copy(tmpPtr, byteAry, 0, size)
        Marshal.DestroyStructure(tmpPtr, typeof<'S>)
        Marshal.FreeHGlobal(tmpPtr)

        byteAry

    /// Verifies that the specified generic type is not obj or IComparable.
    [<RequiresExplicitTypeArguments>]
    let checkProperType<'T> () =
        if typeof<'T> = typeof<obj> || typeof<'T> = typeof<IComparable> then
            failwith "the type must be instantiated with explicit generic parameters"

    /// Returns "Some key" when a key was pressed, otherwise "None".
    let getKey () =
        try
            if Console.KeyAvailable then Some (Console.ReadKey().KeyChar)
            else None
        with :? InvalidOperationException -> 
            // InvalidOperationException is thrown when process does not have a console or 
            // input is redirected from a file.
            None

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

/// Permutation utilities
module Permutation =
    
    /// true if the given list is a permutation of the numbers 0 to perm.Length-1
    let is (perm: int list) =
        let nd = perm.Length
        Set perm = Set [0 .. nd-1]

    let private check (perm: int list) =
        if not (is perm) then
            failwithf "%A is not a permutation" perm

    /// the length of the given permutation
    let length (perm: int list) =
        check perm
        perm.Length

    /// true if then given permutation is the identity permutation
    let isIdentity (perm: int list) =
        check perm
        perm = [0 .. (length perm)-1]

    /// inverts the given permutation
    let invert (perm: int list) =
        check perm
        List.indexed perm
        |> List.sortBy (fun (i, p) -> p)
        |> List.map (fun (i, p) -> i)
    
    /// returns the permutation that would result in applying perm1 after perm2    
    let chain (perm1: int list) (perm2: int list) =
        check perm1
        check perm2
        perm2 |> List.permute (fun i -> perm1.[i])

    /// permutes the list using the given permutation
    let apply (perm: int list) lst =
        check perm
        lst |> List.permute (fun i -> perm.[i])

