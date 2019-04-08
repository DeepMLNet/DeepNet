namespace DeepNet.Utils

open System
open System.Reflection
open System.IO
open System.Runtime.InteropServices
open System.Collections.Concurrent
open FSharp.Reflection



/// List extensions
module internal List =

    /// sets element with index elem to given value
    let rec set elem value lst =
        match lst, elem with
            | l::ls, 0 -> value::ls
            | l::ls, _ -> l::(set (elem-1) value ls)
            | [], _ -> invalidArg "elem" "element index out of bounds"

    /// swaps the elements at the specified positions
    let swap elem1 elem2 lst =
        lst
        |> set elem1 lst.[elem2]
        |> set elem2 lst.[elem1]

    /// removes element with index elem 
    let without elem lst =
        List.concat [List.take elem lst; List.skip (elem+1) lst] 

    /// removes all elements with the given value
    let withoutValue value lst =
        lst |> List.filter ((<>) value)

    /// insert the specified value at index elem
    let insert elem value lst =
        List.concat [List.take elem lst; [value]; List.skip elem lst]

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


/// Map extensions
module internal Map = 

    /// adds all items from q to p
    let join (p:Map<'a,'b>) (q:Map<'a,'b>) = 
        Map(Seq.concat [(Map.toSeq p); (Map.toSeq q)])    

    /// Joins all maps.
    let joinMany (ps: Map<'a, 'b> seq) =
        (Map.empty, ps)
        ||> Seq.fold join

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

    /// Maps the key and value.
    let mapKeyValue fn p =
        p
        |> Map.toSeq
        |> Seq.map (fun (k, v) -> fn k v)
        |> Map.ofSeq

    /// Set of all keys contained in the map.
    let keys m =
        m
        |> Map.toSeq
        |> Seq.map fst
        |> Set.ofSeq

    /// Set of all values contained in the map.
    let values m =
        m
        |> Map.toSeq
        |> Seq.map snd
        |> Set.ofSeq


/// Functions for working with permutations.
module internal Permutation =
    
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
                

/// Extension methods for common collection types.
[<AutoOpen>]
module internal CollectionExtensions =

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

        member this.GetOrAdd key createFn =
            match this.TryFind key with
            | Some v -> v
            | None ->
                let v = createFn key
                this.[key] <- v
                v

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

    // allow access to common collections without having to open System.Collections
    type IReadOnlyDictionary<'TKey, 'TValue> = System.Collections.Generic.IReadOnlyDictionary<'TKey, 'TValue>
    type IReadOnlyCollection<'T> = System.Collections.Generic.IReadOnlyCollection<'T>
    type Dictionary<'TKey, 'TValue> = System.Collections.Generic.Dictionary<'TKey, 'TValue>
    type ConcurrentDictionary<'TKey, 'TValue> = System.Collections.Concurrent.ConcurrentDictionary<'TKey, 'TValue>
    type ConcurrentQueue<'T> = System.Collections.Concurrent.ConcurrentQueue<'T>
    type HashSet<'T> = System.Collections.Generic.HashSet<'T>
    type Queue<'T> = System.Collections.Generic.Queue<'T>
    type LinkedList<'T> = System.Collections.Generic.LinkedList<'T>
    type LinkedListNode<'T> = System.Collections.Generic.LinkedListNode<'T>
    type ConditionalWeakTable<'T, 'K when 'T: not struct and 'K: not struct> = 
        System.Runtime.CompilerServices.ConditionalWeakTable<'T, 'K>



/// Extensions to System.Random.
[<AutoOpen>]
module internal RandomExtensions = 

    type System.Random with

        /// Generates an infinite sequence of non-negative random integers.
        member this.Seq () =
            Seq.initInfinite (fun _ -> this.Next())    

        /// Generates an infinite sequence of non-negative random integers that is less than the specified maximum.
        member this.Seq (maxValue) =
            Seq.initInfinite (fun _ -> this.Next(maxValue))    

        /// Generates an infinite sequence of random integers within the given range.
        member this.Seq (minValue, maxValue) =
            Seq.initInfinite (fun _ -> this.Next(minValue, maxValue))

        /// Generates a random floating-point number within the given range.
        member this.NextDouble (minValue, maxValue) =
            this.NextDouble() * (maxValue - minValue) + minValue

        /// Generates an infinite sequence of random numbers between 0.0 and 1.0.
        member this.SeqDouble () =
            Seq.initInfinite (fun _ -> this.NextDouble())

        /// Generates an infinite sequence of random numbers within the given range.
        member this.SeqDouble (minValue, maxValue) =
            Seq.initInfinite (fun _ -> this.NextDouble(minValue, maxValue))
        
        /// Generates a sample from a normal distribution with the given mean and variance.
        member this.NextNormal (mean, variance) =
            let rec notZeroRnd () =
                match this.NextDouble() with
                | x when x > 0.0 -> x
                | _ -> notZeroRnd()
            let u1, u2 = notZeroRnd(), this.NextDouble()
            let z0 = sqrt (-2.0 * log u1) * cos (2.0 * Math.PI * u2)
            mean + z0 * sqrt variance

        /// Generates an infinite sequence of samples from a normal distribution with the given mean and variance.
        member this.SeqNormal (mean, variance) =
            Seq.initInfinite (fun _ -> this.NextNormal(mean, variance))
        


/// Helper functions for basic type information and conversion.
[<AutoOpen>]
module Primitives =

    let private primitiveTypes = 
        [typeof<byte>; typeof<sbyte>; typeof<int16>; typeof<uint16>;
         typeof<int32>; typeof<uint32>; typeof<int64>; typeof<uint64>;
         typeof<single>; typeof<double>]

    /// Convert given value to specified type and return as obj.
    let convTo (toType: System.Type) (value: obj) =
        let fromType = value.GetType()
        if primitiveTypes |> List.contains toType &&
           primitiveTypes |> List.contains fromType then
            Convert.ChangeType(value, toType)
        else
            let fms = fromType.GetMethods(BindingFlags.Static ||| BindingFlags.Public)
            match fms |> Array.tryFind (fun m -> m.Name = "op_Explicit" 
                                              && m.ReturnType = toType) with
            | Some m -> m.Invoke(null, [|value|])
            | None ->
                match toType.GetMethod("op_Implicit", BindingFlags.Static ||| BindingFlags.Public,
                                       null, [|fromType|], null) with
                | null -> 
                    failwithf "no conversion possible from type %s to type %s"
                              fromType.Name toType.Name
                | m -> m.Invoke(null, [|value|])

    /// Convert given value to type 'T.
    let conv<'T> value : 'T =
        convTo typeof<'T> (box value) :?> 'T

    let private getStaticProperty (typ: Type) name =
        match typ.GetProperty(name, BindingFlags.Public ||| BindingFlags.Static,
                              null, typ, [||], [||]) with
        | null -> 
                failwithf "the type %s must implement the static property %s" 
                          typ.Name name
        | p -> p.GetValue(null)

    /// Zero value for the specifed data type.
    let zeroOf dataType =
        match dataType with
        | t when t=typeof<byte>   -> LanguagePrimitives.GenericZero<byte> |> box
        | t when t=typeof<sbyte>  -> LanguagePrimitives.GenericZero<sbyte> |> box
        | t when t=typeof<int16>  -> LanguagePrimitives.GenericZero<int16> |> box
        | t when t=typeof<uint16> -> LanguagePrimitives.GenericZero<uint16> |> box
        | t when t=typeof<int32>  -> LanguagePrimitives.GenericZero<int32> |> box
        | t when t=typeof<uint32> -> LanguagePrimitives.GenericZero<uint32> |> box
        | t when t=typeof<int64>  -> LanguagePrimitives.GenericZero<int64> |> box
        | t when t=typeof<uint64> -> LanguagePrimitives.GenericZero<uint64> |> box
        | t when t=typeof<single> -> LanguagePrimitives.GenericZero<single> |> box
        | t when t=typeof<double> -> LanguagePrimitives.GenericZero<double> |> box
        | t -> getStaticProperty t "Zero"

    /// Zero value for type 'T.
    let zero<'T> : 'T =
        zeroOf typeof<'T> |> unbox

    /// One value for the specifed data type.
    let oneOf dataType =
        match dataType with
        | t when t=typeof<byte>   -> LanguagePrimitives.GenericOne<byte> |> box
        | t when t=typeof<sbyte>  -> LanguagePrimitives.GenericOne<sbyte> |> box
        | t when t=typeof<int16>  -> LanguagePrimitives.GenericOne<int16> |> box
        | t when t=typeof<uint16> -> LanguagePrimitives.GenericOne<uint16> |> box
        | t when t=typeof<int32>  -> LanguagePrimitives.GenericOne<int32> |> box
        | t when t=typeof<uint32> -> LanguagePrimitives.GenericOne<uint32> |> box
        | t when t=typeof<int64>  -> LanguagePrimitives.GenericOne<int64> |> box
        | t when t=typeof<uint64> -> LanguagePrimitives.GenericOne<uint64> |> box
        | t when t=typeof<single> -> LanguagePrimitives.GenericOne<single> |> box
        | t when t=typeof<double> -> LanguagePrimitives.GenericOne<double> |> box
        | t -> getStaticProperty t "One"

    /// One value for type 'T.
    let one<'T> : 'T =
        oneOf typeof<'T> |> unbox

    /// Minimum value for the specifed numeric data type.
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
        | t -> getStaticProperty t "MinValue"

    /// Minimum value for numeric type 'T.
    let minValue<'T> : 'T = 
        minValueOf typeof<'T> |> unbox

    /// Maximum value for the specified numeric data type.
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
        | t -> getStaticProperty t "MaxValue"

    /// Maximum value for numeric type 'T.
    let maxValue<'T> : 'T = 
        maxValueOf typeof<'T> |> unbox

    /// Size of 'T as int64.
    let sizeof64<'T> =
        int64 sizeof<'T>



[<AutoOpen>]
module internal Generic =        

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

        let args = 
            try FSharpValue.GetTupleFields args
            with :? ArgumentException -> [|args|]
        m.Invoke (instance, args) :?> 'R       

    /// Calls the specified static method on the type 'U with the specified generic type arguments
    /// and the specified arguments in tupled form. Return value is of type 'R.
    let callGeneric<'U, 'R> (methodName: string) (genericTypeArgs: System.Type list) args =
        callGenericInst<'U, 'R> null methodName genericTypeArgs args


[<AutoOpen>]
module internal Generic2 =
    
    type private GenericTypeDesc = {
        GenericType:   string    
        TypeArgs:      string list
    }

    let private cache = ConcurrentDictionary<GenericTypeDesc, obj> ()

    let Generic<'G, 'I> (genericTypeArgs: System.Type list) : 'I =
        let desc = { 
            GenericType=typedefof<'G>.AssemblyQualifiedName
            TypeArgs=genericTypeArgs |> List.map (fun t -> t.AssemblyQualifiedName)
        }
        let inst = cache.GetOrAdd(desc, fun _ -> 
            let typ = typedefof<'G>.MakeGenericType(Array.ofList genericTypeArgs)
            Activator.CreateInstance(typ))
        unbox inst


/// Utility functions
module internal Util =
 
    /// path to application directory under AppData\Local
    let localAppData progName =  
        let lad = System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData)
        System.IO.Path.Combine (lad, progName)
    
    /// Verifies that the specified generic type is not obj or IComparable.
    [<RequiresExplicitTypeArguments>]
    let checkProperType<'T> () =
        if typeof<'T> = typeof<obj> || typeof<'T> = typeof<IComparable> then
            failwith "the type must be instantiated with explicit generic parameters"

    /// Path to this assembly.
    let assemblyPath = 
        let myPath = new Uri(Assembly.GetExecutingAssembly().CodeBase)        
        Uri.UnescapeDataString myPath.AbsolutePath

    /// Directory of this assembly.
    let assemblyDir =
        Path.GetDirectoryName assemblyPath

    /// matches integral values (e.g. 2, 2.0 or 2.0f, etc.)
    let (|Integral|_|) (x: obj) =
        match x.GetType() with
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

    /// Returns "Some key" when a key was pressed, otherwise "None".
    let getKey () =
        try
            if Console.KeyAvailable then Some (Console.ReadKey().KeyChar)
            else None
        with :? InvalidOperationException -> 
            // InvalidOperationException is thrown when process does not have a console or 
            // input is redirected from a file.
            None



/// Utility types and operators.        
[<AutoOpen>]
module internal UtilTypes = 

    /// Default value for options. Returns b if a is None, else the value of a.
    let inline (|?) (a: 'a option) b = 
        match a with
        | Some v -> v
        | None -> b
                      
    // Allow access to DateTime without opening System.
    type DateTime = System.DateTime


/// String extensions
module internal String =

    /// combines sequence of string with given seperator but returns empty if sequence is empty
    let concatButIfEmpty empty sep items =
        if Seq.isEmpty items then empty
        else String.concat sep items

    /// object x converted to a string and capped to a maximum length
    let inline truncObj x =
        let maxLen = 80
        let s = sprintf "%A" x
        let s = s.Replace ("\n", " ")
        if String.length s > maxLen then s.[0..maxLen-3-1] + "..."
        else s

    /// part of a limited string length format specifier
    type LimitedPart =
        /// a delimiter that is always inserted
        | Delim of string
        /// a formatter that is replaced by "..." if string becomes too long
        | Formatter of (int -> string)

    /// builds a string of approximate maximum length from the given format specifier parts
    let limited maxLength parts =
        ("", parts)
        ||> List.fold (fun s p ->
            match p with
            | Delim d -> s + d
            | Formatter fmtFn -> 
                match maxLength - s.Length with
                | remLength when remLength > 0 -> s + fmtFn remLength
                | _ -> s + "..."
            )

    // active pattern parsing
    let private tryParseWith tryParseFunc = 
        tryParseFunc >> function
            | true, v    -> Some v
            | false, _   -> None

    /// Matches a DateTime.
    let (|DateTime|_|) = tryParseWith System.DateTime.TryParse
    /// Matches an int32.
    let (|Int|_|)      = tryParseWith System.Int32.TryParse
    /// Matches an int64.
    let (|Int64|_|)    = tryParseWith System.Int64.TryParse
    /// Matches a single.
    let (|Single|_|)   = tryParseWith System.Single.TryParse
    /// Matches a double.
    let (|Double|_|)   = tryParseWith System.Double.TryParse
    
    /// Matches a string with the given prefix.
    let (|Prefixed|_|) (prefix: string) (str: string) =
        if str.StartsWith prefix then Some str.[prefix.Length..]
        else None



/// Exception helpers
[<AutoOpen>]
module internal Exception =

    /// Raises an InvalidOperationException
    let inline invalidOp fmt =
        Printf.kprintf (fun msg -> raise (InvalidOperationException msg)) fmt

    /// Raises an InvalidArgumentException
    let inline invalidArg arg fmt =
        Printf.kprintf (fun msg -> invalidArg arg msg) fmt

    /// Raises an IndexOutOfRangeException.
    let inline indexOutOfRange fmt =
        Printf.kprintf (fun msg -> raise (IndexOutOfRangeException msg)) fmt


    
/// A concurrent dictionary containing weak references to its keys and values.
/// The `getKey` function must return the key of the specified value.
/// The `create` function must create a new value for the specified key.
type ConcurrentWeakDict<'K, 'V> when 'K: equality and 'V: not struct 
        (getKey: 'V -> 'K, keyEqual: 'K -> 'K -> bool, create: 'K -> 'V) =

    /// Number of dead references that triggers cleanup.
    let deadLimit = 1024

    let mutex = obj ()
    let mutable store = new Dictionary<int, LinkedList<WeakReference<'V>>> ()
    let mutable toClean = new Queue<int> ()

    let tryGet (hsh: int) (k: 'K) =
        match store.TryFind hsh with
        | Some weakValues ->
            weakValues |> Seq.tryPick (fun wv -> 
                match wv.TryGetTarget() with
                | true, v when keyEqual (getKey v) k -> Some v
                | _ -> None)
        | None -> None

    let insert (hsh: int) (v: 'V) =
        let weakValues = 
            match store.TryFind hsh with
            | Some weakValues -> weakValues
            | None ->
                let weakValues = new LinkedList<WeakReference<'V>> ()
                store.[hsh] <- weakValues
                weakValues
        weakValues.AddLast (WeakReference<'V> v) |> ignore

    let clean () =
        while toClean.Count > 0 do
            let hsh = toClean.Dequeue()        
            match store.TryFind hsh with
            | Some weakValues ->
                let mutable node = weakValues.First
                while node <> null do
                    let next = node.Next
                    match node.Value.TryGetTarget () with
                    | false, _ -> weakValues.Remove node
                    | _ -> ()
                    node <- next
                if weakValues.Count = 0 then
                    store.Remove hsh |> ignore
            | None -> ()

    /// Creates or returns the already existing value for the specified key.
    member this.Item
        with get (k: 'K) =
            let hsh = hash k
            lock mutex (fun () ->
                match tryGet hsh k with
                | Some v -> v
                | None ->
                    let v = create k
                    insert hsh v
                    v
            )

    /// Should be called by the finalizer of the specified value.
    member this.Finalized (v: 'V) =
        let hsh = hash (getKey v)
        let deadCount = 
            lock mutex (fun () ->
                toClean.Enqueue hsh
                toClean.Count
            )
        if deadCount > deadLimit then
            lock mutex (fun () -> clean ())

