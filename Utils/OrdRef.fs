namespace DeepNet.Utils

open System
open MBrace.FsPickler


/// An ordered reference to an object.
/// Implements equality testing and comparision based on the object identity.
/// If two OrdRefs are created for the same object, they will be equal.
[<CustomPickler>]
type OrdRef<'T when 'T: not struct> (value: 'T) =
    
    // We assume that no more than 2^64 objects are created.
    // As of 2016 it takes a PC around 2.5 years to count to 2^64, thus we are probably safe :-)

    static let mutable nextId: uint64 = 0UL
    static let idTable = ConditionalWeakTable<'T, obj> ()

    let id: uint64 =
        idTable.GetValue (value, fun _ -> 
            nextId <- nextId + 1UL
            box nextId)
        |> unbox

    /// The referenced value.
    member this.Value = value
    
    /// An unique id for the referenced value.
    member this.Id = id

    interface IEquatable<OrdRef<'T>> with
        member this.Equals (other: OrdRef<'T>) =
            this.Id = other.Id

    override this.Equals other =
        match other with
        | :? OrdRef<'T> as other -> (this :> IEquatable<_>).Equals other
        | _ -> false

    interface IComparable<OrdRef<'T>> with
        member this.CompareTo (other: OrdRef<'T>) =
            this.Id.CompareTo other.Id

    interface IComparable with
        member this.CompareTo other =
            match other with
            | :? OrdRef<'T> as other -> (this :> IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare %A to %A." (this.GetType()) (other.GetType())

    override this.GetHashCode() =
        (this.Id &&& 0x000000007fffffffUL) |> int

    override this.ToString() = this.Value.ToString()

    static member CreatePickler (resolver: IPicklerResolver) =
        let xp = resolver.Resolve<'T> ()
        let writer (ws : WriteState) (ordRef: OrdRef<'T>) =
            xp.Write ws "value" ordRef.Value
        let reader (rs : ReadState) =
            let value = xp.Read rs "value"
            OrdRef value
        Pickler.FromPrimitives(reader, writer)

