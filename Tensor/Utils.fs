namespace Tensor.Utils

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

    /// sequence counting from given value to infinity
    let countingFrom from = seq {
        let mutable i = from
        while true do
            yield i
            i <- i + 1
    }

    /// sequence counting from zero to infinity
    let counting = countingFrom 0


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

    /// transposes a list of lists
    let rec transpose = function
        | (_::_)::_ as m -> List.map List.head m :: transpose (List.map List.tail m)
        | _ -> []

    /// swaps the elements at the specified positions
    let swap elem1 elem2 lst =
        lst
        |> set elem1 lst.[elem2]
        |> set elem2 lst.[elem1]

    /// Elementwise addition of two list. If one list is shorter, it is padded with zeros.
    let inline addElemwise a b =
        let rec build a b =
            match a, b with
            | av::ra, bv::rb -> (av+bv)::(build ra rb)
            | av::ra, [] -> av::(build ra [])
            | [], bv::rb -> bv::(build [] rb)
            | [], [] -> []
        build a b

    /// Elementwise summation of lists. All lists are padded to the same size with zeros.
    let inline sumElemwise (ls: 'a list seq) =
        ([], ls) ||> Seq.fold addElemwise


module Map = 
    /// adds all items from q to p
    let join (p:Map<'a,'b>) (q:Map<'a,'b>) = 
        Map(Seq.concat [(Map.toSeq p); (Map.toSeq q)])    

    /// merges two maps by adding the values of the same key
    let inline addSameKeys (p:Map<'a,'b>) (q:Map<'a,'b>) = 
        (p, q) ||> Map.fold (fun res key value ->
            match res |> Map.tryFind key with
            | Some oldValue -> res |> Map.add key (oldValue + value)
            | None -> res |> Map.add key value)

    /// merges two maps by summing the values of the same key      
    let inline sum (ps:Map<'a,'b> seq) =
        (Map.empty, ps) ||> Seq.fold addSameKeys

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

    /// object x converted to a string and capped to a maximum length
    let truncObj x =
        let maxLen = 80
        let s = sprintf "%A" x
        let s = s.Replace ("\n", " ")
        if String.length s > maxLen then s.[0..maxLen-3-1] + "..."
        else s


module Array2D =
    /// returns a transposed copy of the matrix
    let transpose m = 
        Array2D.init (Array2D.length2 m) (Array2D.length1 m) (fun y x -> m.[x, y])


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

    /// identity permutation of given size
    let identity (size: int) =
        [0 .. size-1]

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

    /// permutation is a swap of two elements
    let (|Swap|_|) (perm: int list) =
        if is perm then
            let idxPerm = List.indexed perm
            match idxPerm |> List.tryFind (fun (pos, dest) -> pos <> dest) with
            | Some (cand, candDest) when perm.[candDest] = cand &&
                    idxPerm |> List.forall (fun (pos, dest) -> pos=cand || pos=candDest || pos=dest) ->
                Some (cand, candDest)
            | _ -> None
        else None
                



[<AutoOpen>]
module UtilTypes =

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

    type System.Collections.Concurrent.ConcurrentQueue<'T> with
        member this.TryDequeue () =
            let value = ref (Unchecked.defaultof<'T>)
            if this.TryDequeue (value) then Some !value
            else None

    type System.Collections.Generic.IReadOnlyDictionary<'TKey, 'TValue> with
        member this.TryFindReadOnly key =
            let value = ref (Unchecked.defaultof<'TValue>)
            if this.TryGetValue (key, value) then Some !value
            else None

    type System.Collections.Generic.Queue<'T> with
        member this.TryPeek =
            if this.Count > 0 then Some (this.Peek())
            else None

    type Dictionary<'TKey, 'TValue> = System.Collections.Generic.Dictionary<'TKey, 'TValue>
    type IReadOnlyDictionary<'TKey, 'TValue> = System.Collections.Generic.IReadOnlyDictionary<'TKey, 'TValue>
    type ConcurrentDictionary<'TKey, 'TValue> = System.Collections.Concurrent.ConcurrentDictionary<'TKey, 'TValue>
    type ConcurrentQueue<'T> = System.Collections.Concurrent.ConcurrentQueue<'T>
    type HashSet<'T> = System.Collections.Generic.HashSet<'T>
    type Queue<'T> = System.Collections.Generic.Queue<'T>
    type IReadOnlyCollection<'T> = System.Collections.Generic.IReadOnlyCollection<'T>

    /// convert given value to specified type and return as obj
    let convTo (typ: System.Type) value =
        Convert.ChangeType(box value, typ)

    /// convert given value to type 'T
    let conv<'T> value : 'T =
        Convert.ChangeType(box value, typeof<'T>) :?> 'T

    /// minimum value for the specifed numeric data type
    let minValueOf dataType =
        match dataType with
        | t when t=typeof<byte>   -> box System.Byte.MinValue 
        | t when t=typeof<sbyte>  -> box System.SByte.MinValue
        | t when t=typeof<int16>  -> box System.Int16.MinValue
        | t when t=typeof<uint16> -> box System.UInt16.MinValue
        | t when t=typeof<int32>  -> box System.Int32.MinValue
        | t when t=typeof<uint32> -> box System.UInt32.MinValue
        | t when t=typeof<int64>  -> box System.Int64.MinValue
        | t when t=typeof<uint64> -> box System.UInt64.MinValue
        | t when t=typeof<single> -> box System.Single.MinValue
        | t when t=typeof<double> -> box System.Double.MinValue
        | _ -> failwithf "no minimum value defined for type %s" dataType.Name

    /// minimum value for numeric type 'T
    let minValue<'T> : 'T = 
        minValueOf typeof<'T> |> unbox

    /// maximum value for the specified numeric data type
    let maxValueOf dataType =
        match dataType with
        | t when t=typeof<byte>   -> box System.Byte.MaxValue 
        | t when t=typeof<sbyte>  -> box System.SByte.MaxValue
        | t when t=typeof<int16>  -> box System.Int16.MaxValue
        | t when t=typeof<uint16> -> box System.UInt16.MaxValue
        | t when t=typeof<int32>  -> box System.Int32.MaxValue
        | t when t=typeof<uint32> -> box System.UInt32.MaxValue
        | t when t=typeof<int64>  -> box System.Int64.MaxValue
        | t when t=typeof<uint64> -> box System.UInt64.MaxValue
        | t when t=typeof<single> -> box System.Single.MaxValue
        | t when t=typeof<double> -> box System.Double.MaxValue
        | _ -> failwithf "no maximum value defined for type %s" dataType.Name

    /// maximum value for numeric type 'T
    let maxValue<'T> : 'T = 
        maxValueOf typeof<'T> |> unbox

    /// size of 'T as int64
    let sizeof64<'T> =
        int64 sizeof<'T>

    /// Default value for options. Returns b if a is None, else the value of a.
    let inline (|?) (a: 'a option) b = 
        match a with
        | Some v -> v
        | None -> b

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
                let gm = typeof<'U>.GetMethod (methodName, BindingFlags.Public ||| BindingFlags.NonPublic ||| 
                                                           BindingFlags.Static ||| BindingFlags.Instance)
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


/// Utility functions
module Util =

    /// all BindingFlags, i.e. public and non-public, static and instance
    let allBindingFlags = 
        BindingFlags.Public ||| BindingFlags.NonPublic ||| 
        BindingFlags.Static ||| BindingFlags.Instance

    /// Compares two objects of possibly different types.
    let compareObjs (this: 'A when 'A :> System.IComparable<'A>) (other: obj) =
        if this.GetType() = other.GetType() then
            (this :> System.IComparable<'A>).CompareTo (other :?> 'A)
        else compare (this.GetType().FullName) (other.GetType().FullName)

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
    let localAppData progName =  
        let lad = System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData)
        System.IO.Path.Combine (lad, progName)
    
    [<Flags>]
    type ErrorModes = 
        | SYSTEM_DEFAULT = 0x0
        | SEM_FAILCRITICALERRORS = 0x0001
        | SEM_NOALIGNMENTFAULTEXCEPT = 0x0004
        | SEM_NOGPFAULTERRORBOX = 0x0002
        | SEM_NOOPENFILEERRORBOX = 0x8000

    [<DllImport("kernel32.dll")>]
    extern ErrorModes private SetErrorMode(ErrorModes mode)

    /// disables the Windows WER dialog box on crash of this application
    let disableCrashDialog () =
        SetErrorMode(ErrorModes.SEM_NOGPFAULTERRORBOX |||
                     ErrorModes.SEM_FAILCRITICALERRORS |||
                     ErrorModes.SEM_NOOPENFILEERRORBOX)
        |> ignore

    /// C++ data type for given type instance
    let cppTypeInst (typ: System.Type) = 
        match typ with
        | _ when typ = typeof<single>    -> "float"
        | _ when typ = typeof<double>    -> "double"
        | _ when typ = typeof<sbyte>     -> "int8_t"
        | _ when typ = typeof<byte>      -> "uint8_t"
        | _ when typ = typeof<int32>     -> "int32_t"
        | _ when typ = typeof<uint32>    -> "uint32_t"
        | _ when typ = typeof<int64>     -> "int64_t"
        | _ when typ = typeof<uint64>    -> "uint64_t"
        | _ when typ = typeof<bool>      -> "bool"
        | _ when typ = typeof<nativeint> -> "ptr_t"
        | _ -> failwithf "no C++ datatype for %A" typ

    /// C++ data type for given type 
    let cppType<'T> = cppTypeInst typeof<'T>

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



        

