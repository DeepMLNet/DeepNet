namespace Tensor.Utils

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

    /// removes element with index elem 
    let without elem lst =
        List.concat [List.take elem lst; List.skip (elem+1) lst] 

    /// insert the specified value at index elem
    let insert elem value lst =
        List.concat [List.take elem lst; [value]; List.skip elem lst]



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

    // allow access to common collections without having to open System.Collections
    type IReadOnlyDictionary<'TKey, 'TValue> = System.Collections.Generic.IReadOnlyDictionary<'TKey, 'TValue>
    type IReadOnlyCollection<'T> = System.Collections.Generic.IReadOnlyCollection<'T>
    type Dictionary<'TKey, 'TValue> = System.Collections.Generic.Dictionary<'TKey, 'TValue>
    type ConcurrentDictionary<'TKey, 'TValue> = System.Collections.Concurrent.ConcurrentDictionary<'TKey, 'TValue>
    type ConcurrentQueue<'T> = System.Collections.Concurrent.ConcurrentQueue<'T>
    type HashSet<'T> = System.Collections.Generic.HashSet<'T>
    type Queue<'T> = System.Collections.Generic.Queue<'T>



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

        let args = FSharpValue.GetTupleFields args
        m.Invoke (instance, args) :?> 'R       

    /// Calls the specified static method on the type 'U with the specified generic type arguments
    /// and the specified arguments in tupled form. Return value is of type 'R.
    let callGeneric<'U, 'R> (methodName: string) (genericTypeArgs: System.Type list) args =
        callGenericInst<'U, 'R> null methodName genericTypeArgs args



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


