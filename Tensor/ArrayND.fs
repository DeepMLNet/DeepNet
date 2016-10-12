namespace ArrayNDNS

open System.Collections
open System.Collections.Generic

open Basics


[<AutoOpen>]
module ArrayNDTypes =

    /// Array storage location
    [<StructuredFormatDisplay("{Pretty}")>]
    type ArrayLocT = 
        | ArrayLoc of string
        with 
            member this.Pretty = 
                let (ArrayLoc loc) = this
                loc

    /// variable stored on host
    let LocHost = ArrayLoc "Host"

    let (|LocHost|_|) arg =
        if arg = ArrayLoc "Host" then Some () else None

    /// raises an error about an unsupported location
    let unsupLoc loc =
        failwithf "location %A is unsupported for this operation" loc

    /// singular matrix encountered
    exception SingularMatrixError of string

    /// ArrayND of any type
    type IArrayNDT =
        abstract Layout:            ArrayNDLayoutT
        abstract Shape:             int list
        abstract NDims:             int
        abstract NElems:            int
        abstract CPPType:           string
        abstract NewView:           ArrayNDLayoutT -> IArrayNDT
        abstract NewOfSameType:     ArrayNDLayoutT -> IArrayNDT
        abstract NewOfType:         ArrayNDLayoutT -> System.Type -> IArrayNDT
        abstract DataType:          System.Type
        abstract Location:          ArrayLocT
        abstract Copy:              unit -> IArrayNDT
        abstract CopyTo:            IArrayNDT -> unit
        abstract GetSlice:          [<System.ParamArray>] args: obj [] -> IArrayNDT
        abstract SetSlice:          [<System.ParamArray>] args: obj [] -> unit
        abstract Item:              [<System.ParamArray>] allArgs: obj [] -> IArrayNDT with get
        abstract Item:              obj -> IArrayNDT with set
        abstract Item:              obj * obj -> IArrayNDT with set
        abstract Item:              obj * obj * obj -> IArrayNDT with set
        abstract Item:              obj * obj * obj * obj -> IArrayNDT with set
        abstract Item:              obj * obj * obj * obj * obj -> IArrayNDT with set
        abstract Item:              obj * obj * obj * obj * obj * obj -> IArrayNDT with set
        abstract Item:              obj * obj * obj * obj * obj * obj * obj -> IArrayNDT with set


    type SpecialAxisT =
        | NewAxis
        | Fill


module ArrayND =

    /// true if warning about fallback copy was shown
    let mutable internal SlowCopyWarningShown = false

    /// an N-dimensional array with reshape and subview abilities
    [<AbstractClass>]
    [<StructuredFormatDisplay("{Pretty}")>]
    type ArrayNDT<'T> (layout: ArrayNDLayoutT) =
        do ArrayNDLayout.check layout

        /// layout
        member this.Layout = layout

        /// value zero (if defined for 'T)
        static member Zero =
            if typeof<'T>.Equals(typeof<double>) then (box 0.0) :?> 'T
            elif typeof<'T>.Equals(typeof<single>) then (box 0.0f) :?> 'T
            elif typeof<'T>.Equals(typeof<int>) then (box 0) :?> 'T
            elif typeof<'T>.Equals(typeof<byte>) then (box 0) :?> 'T
            else failwithf "zero is undefined for type %A" typeof<'T>

        /// value one (if defined for 'T)
        static member One =
            if typeof<'T>.Equals(typeof<double>) then (box 1.0) :?> 'T
            elif typeof<'T>.Equals(typeof<single>) then (box 1.0f) :?> 'T
            elif typeof<'T>.Equals(typeof<int>) then (box 1) :?> 'T
            elif typeof<'T>.Equals(typeof<byte>) then (box 1) :?> 'T
            else failwithf "one is undefined for type %A" typeof<'T>

        /// item access
        abstract Item : int list -> 'T with get, set

        /// a new ArrayND of same type and new storage allocation for given layout
        abstract NewOfSameType : ArrayNDLayoutT -> ArrayNDT<'T>

        /// a new ArrayND of given type and new storage allocation for given layout
        abstract NewOfType<'N> : ArrayNDLayoutT -> ArrayNDT<'N>

        /// a new ArrayND of same type with same storage allocation but new layout
        abstract NewView : ArrayNDLayoutT -> ArrayNDT<'T>

        /// C++ type name
        member this.CPPType = 
            let dims = ArrayNDLayout.nDims layout
            let shp = ArrayNDLayout.shape layout
            let str = ArrayNDLayout.stride layout
            let ofst = ArrayNDLayout.offset layout
            let cppDataType = Util.cppType this.DataType
            let shapeStr = 
                if dims = 0 then "" 
                else "<" + (shp |> Util.intToStrSeq |> String.concat ",") + ">"
            let strideStr = 
                "<" + ((ofst :: str) |> Util.intToStrSeq |> String.concat ",") + ">"
            sprintf "ArrayND%dD<%s, ShapeStatic%dD%s, StrideStatic%dD%s>" 
                dims cppDataType dims shapeStr dims strideStr            

        /// type of data in this ArrayND
        abstract DataType: System.Type
        default this.DataType = typeof<'T>

        /// storage location of the ArrayND
        abstract Location: ArrayLocT

        /// unchecked cast to NDArrayT<'A>
        member this.Cast<'A> () =
            let thisBoxed = box this
            let thisCasted = unbox<ArrayNDT<'A>> thisBoxed
            thisCasted

        /// unchecked cast of v to NDArrayT<'T> (this type)
        member this.CastToMe (v: ArrayNDT<'A>) = v.Cast<'T> ()

        /// checks that two ArrayNDs have the same shape
        static member inline CheckSameShape (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
            if (ArrayNDLayout.shape a.Layout) <> (ArrayNDLayout.shape b.Layout) then
                failwithf "ArrayNDs of shapes %A and %A were expected to have same shape" 
                    (ArrayNDLayout.shape a.Layout) (ArrayNDLayout.shape b.Layout)

        /// Copy the elements of this ArrayNDT to the specified destination ArrayNDT.
        /// Both ArrayNDTs must be of same shape.
        abstract CopyTo : ArrayNDT<'T> -> unit
        default this.CopyTo (dest: ArrayNDT<'T>) =
            // slow element-wise fallback copy
            if not SlowCopyWarningShown then
                printfn "WARNING: fallback slow ArrayNDT.CopyTo is being used \
                         (this message is only shown once)"
                SlowCopyWarningShown <- true
            ArrayNDT<'T>.CheckSameShape this dest
            for idx in ArrayNDLayout.allIdx this.Layout do
                dest.[idx] <- this.[idx]

        /// a view of this ArrayNDT over the given range 
        member this.View rng =
            this.NewView (ArrayNDLayout.view rng this.Layout)

        /// shape
        member this.Shape = ArrayNDLayout.shape this.Layout

        /// number of dimensions
        member this.NDims = ArrayNDLayout.nDims this.Layout

        /// number of elements
        member this.NElems = ArrayNDLayout.nElems this.Layout

        /// broadcasts this and other to the same shape if possible
        member this.BroadcastToSame (other: ArrayNDT<_>) =
            let lThis, lOther = ArrayNDLayout.broadcastToSame this.Layout other.Layout
            this.NewView lThis, other.NewView lOther

        /// broadcasts this and other1 and other2 to the same shape if possible
        member this.BroadcastToSame3 (other1: ArrayNDT<_>) (other2: ArrayNDT<_>) =
            let layouts = ArrayNDLayout.broadcastToSameMany [this.Layout; other1.Layout; other2.Layout]
            match layouts with
            | [lThis; lOther1; lOther2] ->
                this.NewView lThis, other1.NewView lOther1, other2.NewView lOther2
            | _ -> failwith "impossible"

        /// broadcast the list of arrays to the same shape if possible
        static member BroadcastToSameMany (arys: 'A list when 'A :> ArrayNDT<'T>) =
            let layouts = ArrayNDLayout.broadcastToSameMany (arys |> List.map (fun a -> a.Layout))
            List.zip arys layouts |> List.map (fun (a, l) -> a.NewView l :?> 'A)

        /// broadcasts this array to the given shape if possible
        member this.BroadcastToShape shp = 
            let l = ArrayNDLayout.broadcastToShape shp this.Layout
            this.NewView l

        /// implements a storage specific version of map
        abstract MapImpl: ('T -> 'R) -> ArrayNDT<'R> -> unit
        default this.MapImpl f result =
            // slow fallback mapping
            for idx in ArrayNDLayout.allIdx this.Layout do
                result.[idx] <- f this.[idx]

        /// maps all elements using the specified function into a new ArrayNDT
        member this.Map (f: 'T -> 'R) =
            let res = this.NewOfType<'R> (ArrayNDLayout.newC this.Shape)
            this.MapImpl f res
            res

        abstract MapInplaceImpl: ('T -> 'T) -> unit
        default this.MapInplaceImpl f = 
            // slow fallback mapping
            for idx in ArrayNDLayout.allIdx this.Layout do
                this.[idx] <- f this.[idx]

        /// maps all elements using the specified function in-place
        member this.MapInplace (f: 'T -> 'T) =
            this.MapInplaceImpl f

        abstract Map2Impl: ('T -> 'T -> 'R) -> ArrayNDT<'T> -> ArrayNDT<'R> -> unit
        default this.Map2Impl f other result =
            for idx in ArrayNDLayout.allIdx this.Layout do
                result.[idx] <- f this.[idx] other.[idx]

        /// maps all elements of this and other using the specified function into a new ArrayNDT
        member this.Map2 (f: 'T -> 'T -> 'R) (other: #ArrayNDT<'T>) =
            if other.GetType() <> this.GetType() then
                failwithf "cannot use Map2 on ArrayNDTs of different types: %A and %A"
                    (this.GetType()) (other.GetType())
            let this, other = this.BroadcastToSame other
            let res = this.NewOfType<'R> (ArrayNDLayout.newC this.Shape)
            this.Map2Impl f other res
            res

        abstract IfThenElseImpl: ArrayNDT<bool> -> ArrayNDT<'T> -> ArrayNDT<'T> -> unit
        default this.IfThenElseImpl cond elseVal result =
            for idx in ArrayNDLayout.allIdx this.Layout do
                result.[idx] <- if cond.[idx] then this.[idx] else elseVal.[idx]

        /// elementwise uses elements from this if cond is true, 
        /// otherwise elements from elseVal
        member this.IfThenElse (cond: #ArrayNDT<bool>) (elseVal: #ArrayNDT<'T>) =
            if elseVal.GetType() <> this.GetType() then
                failwithf "cannot use IfThenElse on ArrayNDTs of different types: %A and %A"
                    (this.GetType()) (elseVal.GetType())
            if cond.GetType().GetGenericTypeDefinition() <> this.GetType().GetGenericTypeDefinition() then
                failwithf "cannot use IfThenElse on ArrayNDTs of different types: %A and %A"
                    (this.GetType()) (cond.GetType())
            let ifVal, elseVal, cond = this.BroadcastToSame3 elseVal cond
            let res = this.NewOfSameType (ArrayNDLayout.newC ifVal.Shape)
            ifVal.IfThenElseImpl cond elseVal res
            res

        abstract IndexedSetImpl: #ArrayNDT<int> option list -> ArrayNDT<'T> -> unit
        default trgt.IndexedSetImpl indices src =
            for trgtIdx in ArrayNDLayout.allIdx trgt.Layout do
                let srcIdx = 
                    indices 
                    |> List.mapi (fun dim idx ->
                        match idx with
                        | Some di -> di.[trgtIdx]
                        | None -> trgtIdx.[dim])
                trgt.[trgtIdx] <- src.[srcIdx]
     
                       
        /// Sets the values of this array by selecting from the sources array according to the specified
        /// indices. If an index array is set to None then the target index is used as the source index.
        member trgt.IndexedSet (indices: #ArrayNDT<int> option list) (src: #ArrayNDT<'T>) =
            if src.GetType() <> trgt.GetType() then
                failwithf "cannot use IndexedSet on ArrayNDTs of different types: %A and %A"
                    (trgt.GetType()) (src.GetType())
            match indices |> List.tryPick id with
            | Some ih ->
                if ih.GetType().GetGenericTypeDefinition() <> trgt.GetType().GetGenericTypeDefinition() then
                    failwithf "cannot use IndexedSet on ArrayNDTs of different types: %A and %A"
                        (trgt.GetType()) (indices.GetType())
            | None -> ()
            if src.NDims <> indices.Length then
                failwithf "must specify an index array for each dimension of src"
            let indices = indices |> List.map (Option.map (fun idx -> idx.BroadcastToShape trgt.Shape))
            trgt.IndexedSetImpl indices src

        abstract IndexedSumImpl: #ArrayNDT<int> option list -> ArrayNDT<'T> -> unit
        default trgt.IndexedSumImpl indices src = 
            let addInt a b = (a |> box |> unbox<int>) + (b |> box |> unbox<int>) |> box |> unbox<'T>
            let addSingle a b = (a |> box |> unbox<single>) + (b |> box |> unbox<single>) |> box |> unbox<'T>
            let addDouble a b = (a |> box |> unbox<double>) + (b |> box |> unbox<double>) |> box |> unbox<'T>
            let addBool a b = ((a |> box |> unbox<bool>) || (b |> box |> unbox<bool>)) |> box |> unbox<'T>
            let add =
                match typeof<'T> with
                | t when t=typeof<int> -> addInt
                | t when t=typeof<single> -> addSingle
                | t when t=typeof<double> -> addDouble
                | t when t=typeof<bool> -> addBool
                | t -> failwithf "unsupported type: %A" t
            for srcIdx in ArrayNDLayout.allIdx src.Layout do
                let trgtIdx =
                    indices
                    |> List.mapi (fun dim idx ->
                        match idx with
                        | Some di -> di.[srcIdx]
                        | None -> srcIdx.[dim])
                trgt.[trgtIdx] <- add trgt.[trgtIdx] src.[srcIdx]

        /// Sets the values of this array by summing elements from the sources array into the elements
        /// of this array specified by the indices.
        /// If an index array is set to None then the target index is used as the source index.
        member trgt.IndexedSum (indices: #ArrayNDT<int> option list) (src: #ArrayNDT<'T>) =
            if src.GetType() <> trgt.GetType() then
                failwithf "cannot use IndexedSum on ArrayNDTs of different types: %A and %A"
                    (trgt.GetType()) (src.GetType())
            match indices |> List.tryPick id with
            | Some ih ->
                if ih.GetType().GetGenericTypeDefinition() <> trgt.GetType().GetGenericTypeDefinition() then
                    failwithf "cannot use IndexedSum on ArrayNDTs of different types: %A and %A"
                        (trgt.GetType()) (indices.GetType())
                if ih.Shape <> src.Shape then
                    failwithf "index arrays have shapes %A that do not match source shape %A"
                        (indices |> List.map (Option.map (fun a -> a.Shape))) src.Shape
            | None -> ()
            if trgt.NDims <> indices.Length then
                failwithf "must specify an index array for each dimension of the target"
            let indices = indices |> List.map (Option.map (fun idx -> idx.BroadcastToShape src.Shape))
            trgt.IndexedSumImpl indices src

        /// invert the matrix
        abstract Invert : unit -> ArrayNDT<'T>

        // enumerator interfaces
        interface IEnumerable<'T> with
            member this.GetEnumerator() =
                ArrayNDLayout.allIdx this.Layout
                |> Seq.map (fun idx -> this.[idx])
                |> fun s -> s.GetEnumerator()
            member this.GetEnumerator() =
                (this :> IEnumerable<'T>).GetEnumerator() :> IEnumerator

        /// converts .Net item/ranges to RangeT list
        member internal this.ToRng (allArgs: obj []) =
            let rec toRng (args: obj list) =
                match args with
                // direct range specification
                | [:? (RangeT list) as rngs] -> rngs
                // slices
                | (:? (int option) as so) :: (:? (int option) as fo)  :: rest ->
                    Rng (so, fo) :: toRng rest
                // items
                | (:? int as i)           :: rest ->
                    RngElem i :: toRng rest
                | (:? SpecialAxisT as sa) :: rest ->
                    match sa with
                    | NewAxis -> RngNewAxis :: toRng rest
                    | Fill    -> RngAllFill :: toRng rest
                // special cases
                | [] -> []
                | _  -> failwithf "invalid item/slice specification: %A" allArgs 

            allArgs |> Array.toList |> toRng

        member this.GetSlice ([<System.ParamArray>] allArgs: obj []) =
            this.View (this.ToRng allArgs) 

        member this.SetSlice ([<System.ParamArray>] allArgs: obj []) =
            let rngArgs = allArgs.[0 .. allArgs.Length - 2] 
            let trgt = this.View (this.ToRng rngArgs) 
            let valueObj = Array.last allArgs
            match valueObj with
            | :? ArrayNDT<'T> as value -> (value.BroadcastToShape trgt.Shape).CopyTo trgt
            | _ -> failwithf "need array of same type to assign, but got type %A" 
                        (valueObj.GetType())
                
        // item setter does not accept <ParamArray>, thus we have to write it out
        member this.Item
            with get ([<System.ParamArray>] allArgs: obj []) = this.GetSlice (allArgs)
            and set (arg0: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj, arg6: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; arg6; value :> obj|])

        interface IArrayNDT with
            member this.Layout = this.Layout
            member this.CPPType = this.CPPType   
            member this.Shape = this.Shape
            member this.NDims = this.NDims
            member this.NElems = this.NElems      
            member this.NewView layout = this.NewView layout :> IArrayNDT    
            member this.NewOfSameType layout = this.NewOfSameType layout :> IArrayNDT
            member this.NewOfType layout typ = 
                let gm = this.GetType().GetMethod("NewOfType")
                let m = gm.MakeGenericMethod [|typ|]
                m.Invoke(this, [|box layout|]) :?> IArrayNDT
            member this.DataType = this.DataType
            member this.Location = this.Location
            member this.Copy () = 
                let shp = ArrayNDLayout.shape this.Layout
                let trgt = this.NewOfSameType (ArrayNDLayout.newC shp)
                this.CopyTo trgt
                trgt :> IArrayNDT
            member this.CopyTo dest = 
                match dest with
                | :? ArrayNDT<'T> as dest -> this.CopyTo dest
                | _ -> failwith "destination must be of same type as source"
            member this.GetSlice ([<System.ParamArray>] allArgs: obj []) =
                this.GetSlice (allArgs) :> IArrayNDT
            member this.SetSlice ([<System.ParamArray>] allArgs: obj []) =
                this.SetSlice (allArgs)
            member this.Item
                with get ([<System.ParamArray>] allArgs: obj []) = this.GetSlice (allArgs) :> IArrayNDT
                and set (arg0: obj) (value: IArrayNDT) = 
                    this.SetSlice ([|arg0; value :> obj|])
            member this.Item
                with set (arg0: obj, arg1: obj) (value: IArrayNDT) = 
                    this.SetSlice ([|arg0; arg1; value :> obj|])
            member this.Item
                with set (arg0: obj, arg1: obj, arg2: obj) (value: IArrayNDT) = 
                    this.SetSlice ([|arg0; arg1; arg2; value :> obj|])
            member this.Item
                with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj) (value: IArrayNDT) = 
                    this.SetSlice ([|arg0; arg1; arg2; arg3; value :> obj|])
            member this.Item
                with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj) (value: IArrayNDT) = 
                    this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; value :> obj|])
            member this.Item
                with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj) (value: IArrayNDT) = 
                    this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; value :> obj|])
            member this.Item
                with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj, arg6: obj) (value: IArrayNDT) = 
                    this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; arg6; value :> obj|])


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // element access
    ////////////////////////////////////////////////////////////////////////////////////////////////   
    
    /// get element value
    let inline get (idx: int list) (a: ArrayNDT<_>) = a.[idx]
    
    /// set element value
    let inline set (idx: int list) value (a: ArrayNDT<_>) = a.[idx] <- value

    /// if true, then setting NaN or Inf causes and exception to be thrown.
    let CheckFinite = false

    /// checks if value is finite if CheckFinite is true and raises an exception if not
    let inline doCheckFinite value =
        if CheckFinite then
            let isNonFinite =
                match box value with
                | :? double as dv -> System.Double.IsInfinity(dv) || System.Double.IsNaN(dv) 
                | :? single as sv -> System.Single.IsInfinity(sv) || System.Single.IsNaN(sv) 
                | _ -> false
            if isNonFinite then raise (System.ArithmeticException("non-finite value encountered"))

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // shape functions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// location
    let inline location (a: #IArrayNDT) = a.Location

    /// layout
    let inline layout (a: #IArrayNDT) = a.Layout

    /// number of dimensions
    let inline nDims a = layout a |> ArrayNDLayout.nDims

    /// number of elements 
    let inline nElems a = layout a |> ArrayNDLayout.nElems
    
    /// shape in elements
    let inline shape a = layout a |> ArrayNDLayout.shape

    /// stride in elements
    let inline stride a = layout a |> ArrayNDLayout.stride

    /// offset in elements
    let inline offset a = layout a |> ArrayNDLayout.offset

    /// checks that the given axis is valid
    let inline checkAxis ax a = layout a |> ArrayNDLayout.checkAxis ax

    /// sequence of all indices 
    let inline allIdx a = layout a |> ArrayNDLayout.allIdx

    /// all indices of the given dimension
    let inline allIdxOfDim dim a = layout a |> ArrayNDLayout.allIdxOfDim dim 
            
    /// sequence of all elements of a ArrayND
    let inline allElems a = allIdx a |> Seq.map (fun i -> get i a)

    /// true if the ArrayND is contiguous
    let inline isC a = layout a |> ArrayNDLayout.isC

    /// true if the ArrayND is in Fortran order
    let inline isF a = layout a |> ArrayNDLayout.isF

    /// true if the memory of the ArrayND is a contiguous block
    let inline hasContiguousMemory a = layout a |> ArrayNDLayout.hasContiguousMemory

    /// creates a new ArrayND with the same type as passed and contiguous (row-major) layout for specified shape
    let inline newCOfSameType shp (a: 'A when 'A :> IArrayNDT) : 'A =
        a.NewOfSameType (ArrayNDLayout.newC shp) :?> 'A

    /// creates a new ArrayND with the specified type and contiguous (row-major) layout for specified shape
    let inline newCOfType shp (a: 'A when 'A :> ArrayNDT<_>) =
        a.NewOfType (ArrayNDLayout.newC shp) 

    /// creates a new ArrayND with the same type as passed and Fortran (column-major) layout for specified shape
    let inline newFOfSameType shp (a: 'A when 'A :> IArrayNDT) : 'A =
        a.NewOfSameType (ArrayNDLayout.newF shp) :?> 'A

    /// creates a new ArrayND with the specified type and contiguous (column-major) layout for specified shape
    let inline newFOfType shp (a: 'A when 'A :> ArrayNDT<_>) =
        a.NewOfType (ArrayNDLayout.newF shp) 

    /// creates a new ArrayND with existing data but new layout
    let inline relayout newLayout (a: 'A when 'A :> IArrayNDT)  =
        a.NewView newLayout :?> 'A

    /// checks that two ArrayNDs have the same shape
    let inline checkSameShape (a: ArrayNDT<'T>) b =
        ArrayNDT<'T>.CheckSameShape a b

    /// Copies all elements from source to destination.
    /// Both ArrayNDs must have the same shape.
    let inline copyTo (source: #ArrayNDT<'T>) (dest: #ArrayNDT<'T>) =
        source.CopyTo dest

    /// Returns a contiguous copy of the given ArrayND.
    let inline copy source =
        let dest = newCOfSameType (shape source) source
        copyTo source dest
        dest

    /// Returns a contiguous copy of the given IArrayNDT.
    let inline copyUntyped (source: 'T when 'T :> IArrayNDT) =
        source.Copy() :?> 'T

    /// If the ArrayND is not contiguous, returns a contiguous copy; otherwise
    /// the given ArrayND is returned unchanged.
    let inline ensureC a =
        if isC a then a else copy a

    /// makes a contiguous copy of the given tensor if it is not contiguous and with zero offset
    let inline ensureCAndOffsetFree a = 
        if isC a && offset a = 0 then a else copy a 

    /// inserts a broadcastable dimension of size one as first dimension
    let inline padLeft a =
        relayout (ArrayNDLayout.padLeft (layout a)) a

    /// appends a broadcastable dimension of size one as last dimension
    let inline padRight a =
        relayout (ArrayNDLayout.padRight (layout a)) a

    /// cuts one dimension from the left
    let inline cutLeft a =
        relayout (ArrayNDLayout.cutLeft (layout a)) a
      
    /// cuts one dimension from the right
    let inline cutRight a =
        relayout (ArrayNDLayout.cutRight (layout a)) a        

    /// broadcast the given dimension to the given size
    let inline broadcastDim dim size a =
        relayout (ArrayNDLayout.broadcastDim dim size (layout a)) a        

    /// pads shapes from the left until they have same rank
    let inline padToSame a b =
        let la, lb = ArrayNDLayout.padToSame (layout a) (layout b)
        relayout la a, relayout lb b

    /// broadcasts to have the same size
    let inline broadcastToSame a b =
        let la, lb = ArrayNDLayout.broadcastToSame (layout a) (layout b)
        relayout la a, relayout lb b

    /// broadcasts all arrays to have the same shape
    let inline broadcastToSameMany arys =
        ArrayNDT<_>.BroadcastToSameMany arys

    /// broadcasts to have the same size in the given dimensions
    let inline broadcastToSameInDims dims a b =
        let la, lb = ArrayNDLayout.broadcastToSameInDims dims (layout a) (layout b)
        relayout la a, relayout lb b

    /// broadcasts a ArrayND to the given shape
    let inline broadcastToShape shp a =
        relayout (ArrayNDLayout.broadcastToShape shp (layout a)) a

    /// returns true if at least one dimension is broadcasted
    let inline isBroadcasted a =
        ArrayNDLayout.isBroadcasted (layout a)

    /// Reshape array assuming a contiguous (row-major) memory layout.
    /// The current memory layout (as given by the strides) has no influence 
    /// on the reshape operation.
    /// If the array is not contiguous, an error is raised. No copy is performed.
    /// The number of elements must not change.
    let inline reshapeView shp a =
        relayout (ArrayNDLayout.reshape shp (layout a)) a

    /// Reshape array assuming a contiguous (row-major) memory layout.
    /// The current memory layout (as given by the strides) has no influence 
    /// on the reshape operation.
    /// If the array is not contiguous, a reshaped copy is returned.
    /// The number of elements must not change.
    /// One element can be -1, in which case the size of that element is
    /// inferred automatically.
    let inline reshape shp a =
        reshapeView shp (ensureC a)

    /// Flattens the array into a vector assuming a contiguous (row-major) memory layout.
    let inline flatten a =
        reshape [-1] a

    /// swaps the given dimensions
    let inline swapDim ax1 ax2 a =
        relayout (ArrayNDLayout.swapDim ax1 ax2 (layout a)) a

    /// Transposes the given matrix.
    /// If the array has more then two dimensions, the last two axes are swapped.
    let inline transpose a =
        relayout (ArrayNDLayout.transpose (layout a)) a

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    let inline permuteAxes (permut: int list) a =
        a |> relayout (layout a |> ArrayNDLayout.permuteAxes permut)

    /// Reverses the elements in the specified dimension.
    let reverseAxis ax a =
        a |> relayout (layout a |> ArrayNDLayout.reverseAxis ax)        

    /// creates a view of an ArrayND
    let inline view ranges a =
        relayout (ArrayNDLayout.view ranges (layout a)) a        

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // array creation functions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// creates a scalar ArrayND of given value and type
    let scalarOfSameType (a: 'B when 'B :> ArrayNDT<'T>) (value: 'T) : 'B =
        let ary = newCOfSameType [] a
        set [] value ary
        ary

    /// creates a scalar ArrayND of given value 
    let scalarOfType a (value: 'T) =
        let ary = newCOfType [] a
        set [] value ary
        ary

    /// fills the specified ArrayND with zeros
    let inline fillWithZeros (a: #ArrayNDT<'T>) =
        for idx in allIdx a do
            set idx (ArrayNDT<'T>.Zero) a
   
    /// ArrayND of specified shape and same type as a filled with zeros.
    let inline zerosOfSameType shp a =
        newCOfSameType shp a

    /// ArrayND of same shape filled with zeros.
    let inline zerosLike a =
        newCOfSameType (shape a) a

    /// fills with the specified constant
    let fillConst value a =
        for idx in allIdx a do
            a |> set idx value

    /// fills the specified ArrayND with ones
    let fillWithOnes (a: #ArrayNDT<'T>) =
        a |> fillConst ArrayNDT<'T>.One

    /// ArrayND of specified shape and same type as a filled with ones.
    let inline onesOfSameType shp a =
        let n = newCOfSameType shp a
        fillWithOnes n
        n        

    /// ArrayND of same shape filled with ones.
    let inline onesLike a =
        onesOfSameType (shape a) a

    /// fills the diagonal of a quadratic matrix with ones
    let inline fillDiagonalWithOnes (a: #ArrayNDT<'T>) =
        match shape a with
        | [n; m] when n = m ->
            for i = 0 to n - 1 do
                set [i; i] ArrayNDT<'T>.One a
        | _ -> invalidArg "a" "need a quadratic matrix"

    /// Creates a new ArrayNDT by selecting elements from `src` according to the specified `indices`.
    /// `indices` must be a list of ArrayNDTs, one per dimension of `src`. 
    /// If None is specified instead of an array in an dimension, the source index will match the 
    /// target index in that dimension.
    /// The result will have the shape of the (broadcasted) index arrays.
    let select indices (src: #ArrayNDT<'T>) =
        let someIndices = indices |> List.choose id
        if List.isEmpty someIndices then
            failwith "need to specify at least one index array"
        let bcSomeIndices = broadcastToSameMany someIndices
        let rec rebuild idxs repIdxs =
            match idxs, repIdxs with
            | Some idx :: rIdxs, repIdx :: rRepIdxs ->
                Some repIdx :: rebuild rIdxs rRepIdxs
            | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
            | [], [] -> []
            | _ -> failwith "unbalanced idxs"
        let bcIndices = rebuild indices bcSomeIndices
        let trgtShp = bcSomeIndices.Head.Shape
        let trgt = newCOfSameType trgtShp src
        trgt.IndexedSet bcIndices src
        trgt

    /// Creates a new ArrayNDT of shape `trgtShp` by dispersing elements from `src` according to 
    /// the specified `indices`.
    /// `indices` must be a list of ArrayNDTs, one per dimension of `trgt` and of the same shape
    /// (or broadcastable to) as `src`.
    /// If None is specified instead of an array in an dimension, the source index will match the 
    /// target index in that dimension.
    let disperse indices trgtShp (src: #ArrayNDT<'T>) =
        let bcIndices = indices |> List.map (Option.map (broadcastToShape src.Shape))
        let trgt = newCOfSameType trgtShp src
        trgt.IndexedSum bcIndices src
        trgt


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // element-wise operations
    ////////////////////////////////////////////////////////////////////////////////////////////////   
   
    /// Applies the given function element-wise to the given ArrayND and 
    /// stores the result in a new ArrayND.
    let inline map (f: 'T -> 'T) (a: 'A when 'A :> ArrayNDT<'T>) =
        a.Map f :?> 'A

    /// Applies the given function element-wise to the given ArrayND and 
    /// stores the result in a new ArrayND.
    let inline mapTC (f: 'T -> 'R) (a: #ArrayNDT<'T>) =
        a.Map f

    /// Applies the given function element-wise to the given ArrayND inplace.
    let inline mapInplace f (a: #ArrayNDT<'T>) =
        a.MapInplace f

    /// Fills the array with the values returned by the function.
    let inline fill (f: unit -> 'T) (a: #ArrayNDT<'T>) =
        mapInplace (fun _ -> f ()) a

    /// Fills the array with the values returned by the given sequence.
    let fillWithSeq (data: 'T seq) (a: #ArrayNDT<'T>) =
        use enumerator = data.GetEnumerator()
        a |> fill (fun () -> 
            if enumerator.MoveNext() then enumerator.Current
            else failwith "sequence ended before ArrayNDT was filled")

    /// Fills the array with the values returned by the function.
    let inline fillIndexed (f: int list -> 'T) (a: #ArrayNDT<'T>) =
        for idx in allIdx a do
            a.[idx] <- f idx
            
    /// Fills the vector with linearly spaced values from start to (including) stop.
    let inline fillLinSpaced (start: 'T) (stop: 'T) (a: #ArrayNDT<'T>) =
        if a.NDims <> 1 then invalidArg "a" "tensor must be one dimensional"
        if a.NElems < 2 then invalidArg "a" "tensor must have at least two elements"
        let step = (stop - start) / conv<'T> (a.NElems - 1)
        a |> fillIndexed (fun idx -> start + conv<'T> idx.[0] * step)       

    /// Applies the given binary function element-wise to the two given ArrayNDs 
    /// and stores the result in a new ArrayND.
    let inline map2 f (a: 'A when 'A :> ArrayNDT<'T>) (b: 'A) =
        a.Map2 f b :?> 'A

    /// Applies the given binary function element-wise to the two given ArrayNDs 
    /// and stores the result in a new ArrayND.
    let inline map2TC (f: 'T -> 'T -> 'R) (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        a.Map2 f b 

    /// Applies the given binary function element-wise to the two given ArrayNDs 
    /// and stores the result in the first ArrayND.
    let inline map2Inplace f (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        let a, b = broadcastToSame a b
        for idx in allIdx a do
            let cv = f (get idx a) (get idx b)
            set idx cv a

    /// unsupported operation for this type
    let inline unsp (a: 'T) : 'R = 
        failwithf "operation unsupported for type %A" typeof<'T>

   
    let inline uncheckedApply (f: ArrayNDT<'A> -> ArrayNDT<'A>) (a: 'B when 'B :> ArrayNDT<'T>) : 'B =
        let aCast = a.Cast<'A> ()
        let mCast = f aCast
        let m = a.CastToMe mCast
        m :?> 'B

    let inline uncheckedApplyTypeChange (f: ArrayNDT<'A> -> ArrayNDT<'R>) 
            (a: 'B when 'B :> ArrayNDT<'T>) : ArrayNDT<'R> =
        let aCast = a.Cast<'A> ()
        let mCast = f aCast 
        mCast

    let inline uncheckedApply2 (f: ArrayNDT<'A> -> ArrayNDT<'A> -> ArrayNDT<'A>) 
            (a: 'B when 'B :> ArrayNDT<'T>) (b: 'B) : 'B =
        let aCast = a.Cast<'A> ()
        let bCast = b.Cast<'A> ()
        let mCast = f aCast bCast
        let m = a.CastToMe mCast
        m :?> 'B

    let inline uncheckedApply2TypeChange (f: ArrayNDT<'A> -> ArrayNDT<'A> -> ArrayNDT<'R>) 
            (a: 'B when 'B :> ArrayNDT<'T>) (b: 'B) : ArrayNDT<'R> =
        let aCast = a.Cast<'A> ()
        let bCast = b.Cast<'A> ()
        let mCast = f aCast bCast
        mCast

    let inline uncheckedMap (f: 'A -> 'A) (a: #ArrayNDT<'T>) =
        uncheckedApply (map f) a

    let inline uncheckedMap2 (f: 'A -> 'A -> 'A) (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        uncheckedApply2 (map2 f) a b

    let inline uncheckedMap2TypeChange (f: 'A -> 'A -> 'R) (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        uncheckedApply2TypeChange (map2TC f) a b

    let inline typedApply   (fBool:   ArrayNDT<bool>   -> ArrayNDT<bool>) 
                            (fDouble: ArrayNDT<double> -> ArrayNDT<double>) 
                            (fSingle: ArrayNDT<single> -> ArrayNDT<single>)
                            (fInt:    ArrayNDT<int>    -> ArrayNDT<int>)
                            (fByte:   ArrayNDT<byte>   -> ArrayNDT<byte>)
                            (a: #ArrayNDT<'T>) =
        if   typeof<'T>.Equals(typeof<bool>)   then uncheckedApply fBool a 
        elif typeof<'T>.Equals(typeof<double>) then uncheckedApply fDouble a 
        elif typeof<'T>.Equals(typeof<single>) then uncheckedApply fSingle a 
        elif typeof<'T>.Equals(typeof<int>)    then uncheckedApply fInt    a 
        elif typeof<'T>.Equals(typeof<byte>)   then uncheckedApply fByte   a 
        else failwith "unknown type"

    let inline typedApplyTypeChange  (fBool:   ArrayNDT<bool>   -> ArrayNDT<'R>) 
                                     (fDouble: ArrayNDT<double> -> ArrayNDT<'R>) 
                                     (fSingle: ArrayNDT<single> -> ArrayNDT<'R>)
                                     (fInt:    ArrayNDT<int>    -> ArrayNDT<'R>)
                                     (fByte:   ArrayNDT<byte>   -> ArrayNDT<'R>)
                                     (a: #ArrayNDT<'T>) =
        if   typeof<'T>.Equals(typeof<bool>)   then uncheckedApplyTypeChange fBool a 
        elif typeof<'T>.Equals(typeof<double>) then uncheckedApplyTypeChange fDouble a 
        elif typeof<'T>.Equals(typeof<single>) then uncheckedApplyTypeChange fSingle a 
        elif typeof<'T>.Equals(typeof<int>)    then uncheckedApplyTypeChange fInt    a 
        elif typeof<'T>.Equals(typeof<byte>)   then uncheckedApplyTypeChange fByte   a 
        else failwith "unknown type"

    let inline typedApply2  (fBool:   ArrayNDT<bool>   -> ArrayNDT<bool>   -> ArrayNDT<bool>) 
                            (fDouble: ArrayNDT<double> -> ArrayNDT<double> -> ArrayNDT<double>) 
                            (fSingle: ArrayNDT<single> -> ArrayNDT<single> -> ArrayNDT<single>)
                            (fInt:    ArrayNDT<int>    -> ArrayNDT<int>    -> ArrayNDT<int>)
                            (fByte:   ArrayNDT<byte>   -> ArrayNDT<byte>   -> ArrayNDT<byte>)
                            (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        if   typeof<'T>.Equals(typeof<bool>)   then uncheckedApply2 fBool   a b        
        elif typeof<'T>.Equals(typeof<double>) then uncheckedApply2 fDouble a b
        elif typeof<'T>.Equals(typeof<single>) then uncheckedApply2 fSingle a b
        elif typeof<'T>.Equals(typeof<int>)    then uncheckedApply2 fInt    a b
        elif typeof<'T>.Equals(typeof<byte>)   then uncheckedApply2 fByte   a b
        else failwith "unknown type"

    let inline typedApply2TypeChange  (fBool:   ArrayNDT<bool>   -> ArrayNDT<bool>   -> ArrayNDT<'R>) 
                                      (fDouble: ArrayNDT<double> -> ArrayNDT<double> -> ArrayNDT<'R>) 
                                      (fSingle: ArrayNDT<single> -> ArrayNDT<single> -> ArrayNDT<'R>)
                                      (fInt:    ArrayNDT<int>    -> ArrayNDT<int>    -> ArrayNDT<'R>)
                                      (fByte:   ArrayNDT<byte>   -> ArrayNDT<byte>   -> ArrayNDT<'R>)
                                      (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        if   typeof<'T>.Equals(typeof<bool>)   then uncheckedApply2TypeChange fBool   a b
        elif typeof<'T>.Equals(typeof<double>) then uncheckedApply2TypeChange fDouble a b
        elif typeof<'T>.Equals(typeof<single>) then uncheckedApply2TypeChange fSingle a b
        elif typeof<'T>.Equals(typeof<int>)    then uncheckedApply2TypeChange fInt    a b
        elif typeof<'T>.Equals(typeof<byte>)   then uncheckedApply2TypeChange fByte   a b
        else failwith "unknown type"

    let inline typedMap (fBool:   bool   -> bool)
                        (fDouble: double -> double) 
                        (fSingle: single -> single)
                        (fInt:    int    -> int)
                        (fByte:   byte   -> byte)
                        (a: #ArrayNDT<'T>) =
        typedApply (map fBool) (map fDouble) (map fSingle) (map fInt) (map fByte) a

    let inline typedMapTypeChange (fBool:   bool   -> 'R)
                                  (fDouble: double -> 'R) 
                                  (fSingle: single -> 'R)
                                  (fInt:    int    -> 'R)
                                  (fByte:   byte   -> 'R)
                                  (a: #ArrayNDT<'T>) =
        typedApplyTypeChange (mapTC fBool) (mapTC fDouble) (mapTC fSingle) (mapTC fInt) (mapTC fByte) a

    let inline typedMap2 (fBool:   bool   -> bool   -> bool)
                         (fDouble: double -> double -> double) 
                         (fSingle: single -> single -> single)
                         (fInt:    int    -> int    -> int)
                         (fByte:   byte   -> byte   -> byte)
                         (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        typedApply2 (map2 fBool) (map2 fDouble) (map2 fSingle) (map2 fInt) (map2 fByte) a b

    let inline typedMap2TypeChange (fBool:   bool   -> bool   -> 'R)
                                   (fDouble: double -> double -> 'R)
                                   (fSingle: single -> single -> 'R)
                                   (fInt:    int    -> int    -> 'R)
                                   (fByte:   byte   -> byte   -> 'R)
                                   (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        typedApply2TypeChange (map2TC fBool) (map2TC fDouble) (map2TC fSingle) (map2TC fInt) (map2TC fByte) a b

    let inline signImpl (x: 'T) =
        conv<'T> (sign x)

    type ArrayNDT<'T> with    

        // element-wise unary
        static member (~+)      (a: #ArrayNDT<'T>) = typedMap (unsp) (~+) (~+) (~+) (unsp) a
        static member (~-)      (a: #ArrayNDT<'T>) = typedMap (unsp) (~-) (~-) (~-) (unsp) a
        static member Abs       (a: #ArrayNDT<'T>) = typedMap (unsp) abs abs abs (unsp) a
        static member SignT     (a: #ArrayNDT<'T>) = typedMap (unsp) signImpl signImpl sign (unsp) a
        static member Log       (a: #ArrayNDT<'T>) = typedMap (unsp) log log (unsp) (unsp) a
        static member Log10     (a: #ArrayNDT<'T>) = typedMap (unsp) log10 log10 (unsp) (unsp) a
        static member Exp       (a: #ArrayNDT<'T>) = typedMap (unsp) exp exp (unsp) (unsp) a
        static member Sin       (a: #ArrayNDT<'T>) = typedMap (unsp) sin sin (unsp) (unsp) a
        static member Cos       (a: #ArrayNDT<'T>) = typedMap (unsp) cos cos (unsp) (unsp) a
        static member Tan       (a: #ArrayNDT<'T>) = typedMap (unsp) tan tan (unsp) (unsp) a
        static member Asin      (a: #ArrayNDT<'T>) = typedMap (unsp) asin asin (unsp) (unsp) a
        static member Acos      (a: #ArrayNDT<'T>) = typedMap (unsp) acos acos (unsp) (unsp) a
        static member Atan      (a: #ArrayNDT<'T>) = typedMap (unsp) atan atan (unsp) (unsp) a
        static member Sinh      (a: #ArrayNDT<'T>) = typedMap (unsp) sinh sinh (unsp) (unsp) a
        static member Cosh      (a: #ArrayNDT<'T>) = typedMap (unsp) cosh cosh (unsp) (unsp) a
        static member Tanh      (a: #ArrayNDT<'T>) = typedMap (unsp) tanh tanh (unsp) (unsp) a
        static member Sqrt      (a: #ArrayNDT<'T>) = typedMap (unsp) sqrt sqrt (unsp) (unsp) a
        static member Ceiling   (a: #ArrayNDT<'T>) = typedMap (unsp) ceil ceil (unsp) (unsp) a
        static member Floor     (a: #ArrayNDT<'T>) = typedMap (unsp) floor floor (unsp) (unsp) a
        static member Round     (a: #ArrayNDT<'T>) = typedMap (unsp) round round (unsp) (unsp) a
        static member Truncate  (a: #ArrayNDT<'T>) = typedMap (unsp) truncate truncate (unsp) (unsp) a

        // element-wise unary logic
        static member (~~~~)    (a: #ArrayNDT<bool>) = map not a

        // element-wise binary
        static member (+) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (+) (+) (+) (+) a b
        static member (-) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (-) (-) (-) (-) a b
        static member (*) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (*) (*) (*) (*) a b
        static member (/) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (/) (/) (/) (/) a b
        static member (%) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (%) (%) (%) (%) a b
        static member Pow (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) ( ** ) ( ** ) (unsp) (unsp) a b

        // element-wise binary logic
        static member (&&&&) (a: #ArrayNDT<bool>, b: #ArrayNDT<bool>) = map2 (&&) a b
        static member (||||) (a: #ArrayNDT<bool>, b: #ArrayNDT<bool>) = map2 (||) a b

        // element-wise binary comparison
        static member (====) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (=) (=) (=) (=) (=) a b
        static member (<<<<) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (<) (<) (<) (<) (<) a b
        static member (<<==) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (<=) (<=) (<=) (<=) (<=) a b
        static member (>>>>) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (>) (>) (>) (>) (>) a b
        static member (>>==) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (>=) (>=) (>=) (>=) (>=) a b
        static member (<<>>) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (<>) (<>) (<>) (<>) (<>) a b

        // element-wise binary with scalars
        static member inline (+) (a: #ArrayNDT<'T>, b: 'T) = a + (scalarOfSameType a b)
        static member inline (-) (a: #ArrayNDT<'T>, b: 'T) = a - (scalarOfSameType a b)
        static member inline (*) (a: #ArrayNDT<'T>, b: 'T) = a * (scalarOfSameType a b)
        static member inline (/) (a: #ArrayNDT<'T>, b: 'T) = a / (scalarOfSameType a b)
        static member inline (%) (a: #ArrayNDT<'T>, b: 'T) = a % (scalarOfSameType a b)
        static member inline Pow (a: #ArrayNDT<'T>, b: 'T) = a ** (scalarOfSameType a b)        
        static member inline (&&&&) (a: #ArrayNDT<bool>, b: bool) = a &&&& (scalarOfSameType a b)
        static member inline (||||) (a: #ArrayNDT<bool>, b: bool) = a |||| (scalarOfSameType a b)
        static member (====) (a: #ArrayNDT<'T>, b: 'T) = typedMap2TypeChange (=) (=) (=) (=) (=) a (scalarOfSameType a b)   
        static member (<<<<) (a: #ArrayNDT<'T>, b: 'T) = typedMap2TypeChange (<) (<) (<) (<) (<) a (scalarOfSameType a b)   
        static member (<<==) (a: #ArrayNDT<'T>, b: 'T) = typedMap2TypeChange (<=) (<=) (<=) (<=) (<=) a (scalarOfSameType a b)    
        static member (>>>>) (a: #ArrayNDT<'T>, b: 'T) = typedMap2TypeChange (>) (>) (>) (>) (>) a (scalarOfSameType a b)   
        static member (>>==) (a: #ArrayNDT<'T>, b: 'T) = typedMap2TypeChange (>=) (>=) (>=) (>=) (>=) a (scalarOfSameType a b)   
        static member (<<>>) (a: #ArrayNDT<'T>, b: 'T) = typedMap2TypeChange (<>) (<>) (<>) (<>) (<>) a (scalarOfSameType a b)   

        static member inline (+) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfSameType b a) + b
        static member inline (-) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfSameType b a) - b
        static member inline (*) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfSameType b a) * b
        static member inline (/) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfSameType b a) / b
        static member inline (%) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfSameType b a) % b
        static member inline Pow (a: 'T, b: #ArrayNDT<'T>) = (scalarOfSameType b a) ** b
        static member inline (&&&&) (a: bool, b: #ArrayNDT<bool>) = (scalarOfSameType b a) &&&& b
        static member inline (||||) (a: bool, b: #ArrayNDT<bool>) = (scalarOfSameType b a) |||| b
        static member (====) (a: 'T, b: #ArrayNDT<'T>) = typedMap2TypeChange (=) (=) (=) (=) (=) (scalarOfSameType b a) b
        static member (<<<<) (a: 'T, b: #ArrayNDT<'T>) = typedMap2TypeChange (<) (<) (<) (<) (<) (scalarOfSameType b a) b
        static member (<<==) (a: 'T, b: #ArrayNDT<'T>) = typedMap2TypeChange (<=) (<=) (<=) (<=) (<=) (scalarOfSameType b a) b
        static member (>>>>) (a: 'T, b: #ArrayNDT<'T>) = typedMap2TypeChange (>) (>) (>) (>) (>) (scalarOfSameType b a) b
        static member (>>==) (a: 'T, b: #ArrayNDT<'T>) = typedMap2TypeChange (>=) (>=) (>=) (>=) (>=) (scalarOfSameType b a) b
        static member (<<>>) (a: 'T, b: #ArrayNDT<'T>) = typedMap2TypeChange (<>) (<>) (<>) (<>) (<>) (scalarOfSameType b a) b

        // transposition
        member this.T = transpose this

    /// sign keeping type
    let inline signt (a: #ArrayNDT<'T>) =
        ArrayNDT<'T>.SignT a 

    /// Elementwise check if two arrays have same (within machine precision) values.
    /// Check for exact equality when type is int or bool.
    let inline isCloseWithTol (aTol: 'T) (rTol: 'T) (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        match typeof<'T> with
        | t when t = typeof<bool> -> (box a :?> ArrayNDT<bool>) ==== (box b :?> ArrayNDT<bool>) 
        | t when t = typeof<int>  -> (box a :?> ArrayNDT<int>)  ==== (box b :?> ArrayNDT<int>) 
        | _ ->  abs (a - b) <<== aTol + rTol * abs b

    /// Elementwise check if two arrays have same (within machine precision) values.
    let inline isClose (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        isCloseWithTol (conv<'T> 1e-8) (conv<'T> 1e-5) a b

    /// Elementwise check if a value is finite (not NaN and not infinite).
    let inline isFinite (a: ArrayNDT<'T>) =
        let isFiniteSingle v = not (System.Single.IsInfinity v || System.Single.IsNaN v)
        let isFiniteDouble v = not (System.Double.IsInfinity v || System.Double.IsNaN v)
        typedMapTypeChange (unsp) isFiniteDouble isFiniteSingle (unsp) (unsp) a

    /// Elementwise picks the maximum of a or b.
    let inline maxElemwise (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        typedMap2 (max) (max) (max) (max) (max) a b

    /// Elementwise picks the minimum of a or b.
    let inline minElemwise (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        typedMap2 (min) (min) (min) (min) (min) a b

    /// Elementwise uses elements from ifTrue if cond is true, 
    /// otherwise elements from ifFalse.
    let inline ifThenElse (cond: #ArrayNDT<bool>) (ifTrue: 'B when 'B :> ArrayNDT<'T>) (ifFalse: 'B) : 'B =
        ifTrue.IfThenElse cond ifFalse :?> 'B

    /// converts the from one data type to another
    let convert (a: #ArrayNDT<'T>) : ArrayNDT<'C> =
        a |> mapTC (fun v -> conv<'C> v)

    /// converts to int
    let int (a: #ArrayNDT<'T>) : ArrayNDT<int> =
        convert a

    /// converts to float
    let float (a: #ArrayNDT<'T>) : ArrayNDT<float> =
        convert a

    /// converts to single
    let single (a: #ArrayNDT<'T>) : ArrayNDT<single> =
        convert a

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // reduction operations
    ////////////////////////////////////////////////////////////////////////////////////////////////         

    /// value of scalar array
    let inline value a =
        match nDims a with
        | 0 -> get [] a
        | _ -> failwithf "array of shape %A is not a scalar" (shape a)
      
    /// applies the given reduction function over the given dimension
    let inline axisReduceTypeChange (f: ArrayNDT<'T> -> ArrayNDT<'R>) dim (a: ArrayNDT<'T>) : ArrayNDT<'R> =
        let c = newCOfType (List.without dim (shape a)) a
        for srcRng, dstIdx in ArrayNDLayout.allSrcRngsAndTrgtIdxsForAxisReduce dim (layout a) do
            set dstIdx (f (view srcRng a) |> get []) c
        c

    /// applies the given reduction function over the given dimension
    let inline axisReduce (f: ArrayNDT<'T> -> ArrayNDT<'T>) dim (a: 'A when 'A :> ArrayNDT<'T>) : 'A =
        axisReduceTypeChange f dim a :?> 'A

    let inline private sumImpl (a: ArrayNDT<'T>) =
        allElems a 
        |> Seq.fold (+) ArrayNDT<'T>.Zero         
        |> scalarOfSameType a 

    /// element-wise sum
    let sum (a: #ArrayNDT<'T>) =
        typedApply (unsp) sumImpl sumImpl sumImpl sumImpl a 

    /// element-wise sum over given axis
    let sumAxis dim a = 
        axisReduce sum dim a

    /// mean 
    let mean (a: 'A when 'A :> ArrayNDT<'T>) : 'A =
        let a = a :> ArrayNDT<'T>
        sum a / scalarOfSameType a (conv<'T> (nElems a)) :?> 'A

    /// mean over given axis
    let meanAxis dim a = 
        axisReduce mean dim a
    
    let inline private productImpl (a: ArrayNDT<'T>) =
        allElems a 
        |> Seq.fold (*) ArrayNDT<'T>.One
        |> scalarOfSameType a 

    /// element-wise product
    let product (a: #ArrayNDT<'T>) =
        typedApply (unsp) productImpl productImpl productImpl productImpl a 

    /// element-wise product over given axis
    let productAxis dim a = 
        axisReduce product dim a

    let inline private maxImpl a =
        allElems a 
        |> Seq.reduce max
        |> scalarOfSameType a 

    /// maximum value
    let max a =
        if nElems a = 0 then invalidArg "a" "cannot compute max of empty ArrayNDT"
        typedApply (unsp) maxImpl maxImpl maxImpl maxImpl a
    
    /// position of maximum value
    let argMax a =
        allIdx a
        |> Seq.maxBy (fun idx -> a |> get idx)

    /// maximum value over given axis
    let maxAxis dim a = 
        axisReduce max dim a

    let inline private argMaxAxisReduc (a: ArrayNDT<'T>) =
        allIdx a
        |> Seq.maxBy (fun idx -> a |> get idx)
        |> fun idx -> idx.[0]
        |> scalarOfType a

    /// positions of maximum values along given axis
    let argMaxAxis dim (a: ArrayNDT<'T>) : ArrayNDT<int> =
        axisReduceTypeChange argMaxAxisReduc dim a

    let inline private minImpl a =
        allElems a 
        |> Seq.reduce min
        |> scalarOfSameType a 

    /// minimum value
    let min a =
        if nElems a = 0 then invalidArg "a" "cannot compute min of empty ArrayNDT"
        typedApply (unsp) minImpl minImpl minImpl minImpl a

    /// position of maximum value
    let argMin a =
        allIdx a
        |> Seq.minBy (fun idx -> a |> get idx)

    /// minimum value over given axis
    let minAxis dim a = 
        axisReduce min dim a

    let inline private argMinAxisReduc (a: ArrayNDT<'T>) =
        allIdx a
        |> Seq.minBy (fun idx -> a |> get idx)
        |> fun idx -> idx.[0]
        |> scalarOfType a

    /// positions of maximum values along given axis
    let argMinAxis dim (a: ArrayNDT<'T>) : ArrayNDT<int> =
        axisReduceTypeChange argMinAxisReduc dim a

    /// true if all elements of the array are true
    let all a =
        let value = allElems a |> Seq.fold (&&) true
        scalarOfSameType a value

    /// true if any element of the array is true
    let any a =
        let value = allElems a |> Seq.fold (||) false
        scalarOfSameType a value
     
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // tensor operations
    ////////////////////////////////////////////////////////////////////////////////////////////////         

    /// Returns true if two arrays have same (within specified precision) values in all elements.
    /// If arrays have different shape, then false is returned.
    let inline almostEqualWithTol (aTol: 'T) (rTol: 'T) (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        if a.Shape = b.Shape then
            isCloseWithTol aTol rTol a b |> all
        else 
            let res = newCOfType [] a
            set [] false res
            res

    /// Returns true if two arrays have same (within machine precision) values in all elements.
    /// If arrays have different shape, then false is returned.
    let inline almostEqual (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        almostEqualWithTol (conv<'T> 1e-8) (conv<'T> 1e-5) a b

    /// Returns true if all values in the tensor are finite (not NaN and not infinite).
    let inline allFinite (a: ArrayNDT<'T>) =
        a |> isFinite |> all

    /// dot product implementation between vec*vec, mat*vec, mat*mat, batched mat*vec, batched mat*mat
    let inline dotImpl (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        let inline matrixDot a b =
            let nI = (shape a).[0]
            let nJ = (shape a).[1]
            let nK = (shape b).[1]
            let c = newCOfSameType [nI; nK] a
            for k=0 to nK - 1 do
                for i=0 to nI - 1 do
                    let v = 
                        {0 .. nJ - 1}
                        |> Seq.map (fun j -> (get [i; j] a) * (get [j; k] b))
                        |> Seq.sum
                    set [i; k] v c
            c

        let inline batchedMatrixDot (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
            let a, b = broadcastToSameInDims [0..nDims a - 3] a b
            let aRows, aCols = (shape a).[nDims a - 2], (shape a).[nDims a - 1]
            let bRows, bCols = (shape b).[nDims b - 2], (shape b).[nDims b - 1]
            if aCols <> bRows then
                failwithf "cannot compute batched dot product between arrays of shapes %A and %A" 
                    (shape a) (shape b)                
            let smplShape = (shape a).[0 .. nDims a - 3]
            let nSmpls = List.fold (*) 1 smplShape
            let a = reshape [nSmpls; aRows; aCols] a
            let b = reshape [nSmpls; bRows; bCols] b
            let c = newCOfSameType [nSmpls; aRows; bCols] a
            for smpl = 0 to nSmpls - 1 do
                c.[smpl, *, *] <- matrixDot a.[smpl, *, *] b.[smpl, *, *]
            c |> reshape (smplShape @ [aRows; bCols])         

        match nDims a, nDims b with
            | 1, 1 when shape a = shape b -> 
                map2 (*) a b |> sum
            | 2, 1 when (shape a).[1] = (shape b).[0] -> 
                matrixDot a (padRight b) |> view [RngAll; RngElem 0] 
            | 2, 2 when (shape a).[1] = (shape b).[0] ->
                matrixDot a b
            | na, nb when na > 2 && na = nb+1 && (shape a).[na-1] = (shape b).[nb-1] ->
                // batched mat*vec
                (batchedMatrixDot a (padRight b)).[Fill, 0]
            | na, nb when na > 2 && na = nb && (shape a).[na-1] = (shape b).[nb-2] ->
                // batched mat*mat
                batchedMatrixDot a b
            | _ -> 
                failwithf "cannot compute dot product between arrays of shapes %A and %A" 
                    (shape a) (shape b)

    type ArrayNDT<'T> with   
        /// dot product
        static member (.*) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedApply2 (unsp) dotImpl dotImpl dotImpl dotImpl a b

    /// dot product between vec*vec, mat*vec, mat*mat, batched mat*vec, batched mat*mat
    let inline dot a b =
        a .* b

    /// block array specification
    type BlockSpec<'T> =
        | Blocks of BlockSpec<'T> list
        | Array of ArrayNDT<'T>

    /// array constructed of other arrays
    let inline blockArray bs =

        let rec commonShape joinDim shps =               
            match shps with
            | [shp] ->
                List.set joinDim -1 shp
            | shp::rShps ->
                let commonShp = commonShape joinDim [shp]
                if commonShp <> commonShape joinDim rShps then
                    failwithf "block array blocks must have same rank and be identical in all but the join dimension"
                commonShp
            | [] -> []

        let joinSize joinDim (shps: int list list) =
            shps |> List.map (fun shp -> shp.[joinDim]) |> List.sum

        let joinShape joinDim shps =
            commonShape joinDim shps 
                |> List.set joinDim (joinSize joinDim shps)

        let rec joinedBlocksShape joinDim bs =
            match bs with
            | Blocks blcks ->
                blcks |> List.map (joinedBlocksShape (joinDim + 1)) |> joinShape joinDim
            | Array ary ->
                ary |> shape

        let rec blockPosAndContents (joinDim: int) startPos bs = seq {
            match bs with
            | Blocks blcks ->
                let mutable pos = startPos
                for blck in blcks do
                    yield! blockPosAndContents (joinDim + 1) pos blck 
                    let blckShape = joinedBlocksShape joinDim blck
                    pos <- List.set joinDim (pos.[joinDim] + blckShape.[joinDim]) pos
            | Array ary ->
                yield startPos, ary
        }

        let rec anyArray bs =
            match bs with
            | Blocks b -> List.tryPick anyArray b
            | Array a -> Some a
                  
        let tmplArray = Option.get (anyArray bs)
        let joinedShape = joinedBlocksShape 0 bs
        let joined = newCOfSameType joinedShape tmplArray
        let startPos = List.replicate (List.length joinedShape) 0

        for pos, ary in blockPosAndContents 0 startPos bs do
            let slice = List.map2 (fun p s -> Rng(Some p, Some (p + s))) pos (shape ary)
            let joinedSlice = joined |> view slice 
            copyTo ary joinedSlice

        joined
    
    /// tensor product
    let inline tensorProductImpl (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) : ArrayNDT<'T> =
        let a, b = padToSame a b
        let aShp = shape a

        let rec generate pos = 
            match List.length pos with
            | dim when dim = nDims a ->
                let aElem = get pos a
                Array (aElem * b)
            | dim ->
                seq {for p in 0 .. aShp.[dim] - 1 -> generate (pos @ [p])}
                    |> Seq.toList |> Blocks

        generate [] |> blockArray
   
    type ArrayNDT<'T> with
        /// tensor product
        static member (%*) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedApply2 (unsp) tensorProductImpl tensorProductImpl tensorProductImpl tensorProductImpl a b
        
    /// tensor product
    let inline tensorProduct (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) : ArrayNDT<'T> = a %* b

    /// Returns a view of the diagonal along the given axes.
    /// The diagonal replaces the first axis and the second axis is removed.
    let diagAxis ax1 ax2 (a: #ArrayNDT<'T>) =
        relayout (ArrayNDLayout.diagAxis ax1 ax2 a.Layout) a

    /// Returns a view of the diagonal of a matrix as a vector.
    /// If the specified tensor has more than two dimensions, the diagonals
    /// along the last two dimensions are returned.
    let diag (a: #ArrayNDT<'T>) =
        if a.NDims < 2 then
            failwithf "need at least a two dimensional array for diagonal but got shape %A" a.Shape
        diagAxis (a.NDims-2) (a.NDims-1) a

    /// Creates a new array of same shape but with ax2 inserted.
    /// The diagonal over ax1 and ax2 is filled with the elements of the original ax1.
    /// The other elements are set to zero.
    let diagMatAxis ax1 ax2 (a: #ArrayNDT<'T>) =
        if ax1 = ax2 then failwithf "axes to use for diagonal must be different"
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        checkAxis ax1 a
        if not (0 <= ax2 && ax2 <= a.NDims) then
            failwithf "cannot insert axis at position %d into array of shape %A" ax2 a.Shape
        let dShp = a.Shape |> List.insert ax2 a.Shape.[ax1]
        let d = newCOfSameType dShp a
        let dDiag = diagAxis ax1 ax2 d
        dDiag.[Fill] <- a
        d

    /// Creates a new matrix that has the specified diagonal.
    /// All other elements are zero.
    /// If the specified array has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    let diagMat (a: #ArrayNDT<'T>) =
        if a.NDims < 1 then
            failwithf "need at leat a one-dimensional array to create a diagonal matrix"
        diagMatAxis (a.NDims-1) a.NDims a

    /// Computes the traces along the given axes.
    let traceAxis ax1 ax2 (a: #ArrayNDT<'T>) =
        let tax = if ax1 < ax2 then ax1 else ax1 - 1
        a |> diagAxis ax1 ax2 |> sumAxis tax

    /// Computes the trace of a matrix.
    /// If the specified tensor has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    let trace (a: #ArrayNDT<'T>) =
        if a.NDims < 2 then
            failwithf "need at least a two dimensional array for trace but got shape %A" a.Shape
        traceAxis (a.NDims-2) (a.NDims-1) a

    /// Returns the inverse of the given matrix.
    /// If the specified tensor has more than two dimensions, the matrices
    /// consisting of the last two dimensions are inverted.
    let invert (a: 'A when 'A :> ArrayNDT<_>) : 'A  =
        a.Invert () :?> 'A

    /// calculates the pairwise differences along the given axis
    let diffAxis ax (a: #ArrayNDT<'T>) =
        checkAxis ax a
        let shftRng = 
            [for d=0 to a.NDims-1 do
                if d = ax then yield Rng (Some 1, None)
                else yield RngAll]
        let cutRng = 
            [for d=0 to a.NDims-1 do
                if d = ax then yield Rng (None, Some (a.Shape.[d] - 2))
                else yield RngAll]
        a.[shftRng] - a.[cutRng]

    /// calculates the pairwise differences along the last axis
    let diff (a: #ArrayNDT<'T>) =
        diffAxis (a.NDims-1) a

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // concatenation and replication
    ////////////////////////////////////////////////////////////////////////////////////////////////         

    /// Concatenates the list of tensors in the given axis.
    let concat dim (arys: #ArrayNDT<'T> list) =
        if List.isEmpty arys then
            invalidArg "arys" "cannot concatenate empty list of tensors"

        // check for compatibility
        let shp = List.head arys |> shape
        if not (0 <= dim && dim < shp.Length) then
            failwithf "concatenation axis %d is out of range for shape %A" dim shp
        for aryIdx, ary in List.indexed arys do
            if List.without dim ary.Shape <> List.without dim shp then
                failwithf "concatentation element with index %d with shape %A must \
                    be equal to shape %A of the first element, except in the concatenation axis %d" 
                    aryIdx ary.Shape shp dim

        // calculate shape of concatenated tensors
        let totalSize = arys |> List.sumBy (fun ary -> ary.Shape.[dim])
        let concatShape = shp |> List.set dim totalSize

        // copy tensors into concatenated tensor
        let cc = List.head arys |> newCOfSameType concatShape
        let mutable pos = 0
        for ary in arys do
            let aryLen = ary.Shape.[dim]
            if aryLen > 0 then
                let ccRng = 
                    List.init shp.Length (fun idx ->
                        if idx = dim then Rng (Some pos, Some (pos + aryLen - 1))
                        else RngAll)
                cc.[ccRng] <- ary
                pos <- pos + aryLen
        cc

    /// Replicates the tensor the given number of repetitions along the given axis.
    let replicate dim reps (ary: #ArrayNDT<'T>) =
        ary |> checkAxis dim
        if reps < 0 then
            invalidArg "reps" "number of repetitions cannot be negative"

        // 1. insert axis of size one left to repetition axis
        // 2. broadcast along the new axis to number of repetitions
        // 3. reshape to result shape
        ary 
        |> reshape (ary.Shape |> List.insert dim 1)
        |> broadcastDim dim reps
        |> reshape (ary.Shape |> List.set dim (reps * ary.Shape.[dim]))


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // pretty printing
    ////////////////////////////////////////////////////////////////////////////////////////////////         
    
    /// Pretty string containing maxElems elements per dimension.
    /// If maxElems is zero, then the elements per dimension are unlimited.
    let pretty maxElems (a: ArrayNDT<'T>) =
        let maxElems =
            if maxElems > 0 then maxElems
            else Microsoft.FSharp.Core.int.MaxValue

        let rec prettyDim lineSpace a =
            let ls () = (shape a).[0]
            let subPrint idxes = 
                idxes
                |> Seq.map (fun i -> 
                    prettyDim (lineSpace + " ") (view [RngElem i; RngAllFill] a)) 
                |> Seq.toList                   
            let subStrs () = 
                if ls() <= maxElems then
                    subPrint (seq {0 .. ls() - 1})
                else
                    let leftTo = maxElems / 2 - 1
                    let remaining = maxElems - 1 - leftTo - 1
                    let rightFrom = ls() - remaining
                    let leftIdx = seq {0 .. leftTo}
                    let rightIdx = seq {rightFrom .. (ls()-1)}
                    let elipsis =
                        match typeof<'T> with
                        | t when t=typeof<single> -> "      ..."
                        | t when t=typeof<double> -> "      ..."
                        | t when t=typeof<int>    -> " ..."
                        | t when t=typeof<byte>   -> "..."
                        | t when t=typeof<bool>   -> " ... "
                        | _ -> "..."
                    (subPrint leftIdx) @ [elipsis] @ (subPrint rightIdx)

            match nDims a with
            | 0 -> 
                let v = value a
                if   typeof<'T>.Equals(typeof<single>) then sprintf "%9.4f" (v |> box :?> single)
                elif typeof<'T>.Equals(typeof<double>) then sprintf "%9.4f" (v |> box :?> double)
                elif typeof<'T>.Equals(typeof<int>)    then sprintf "%4d"  (v |> box :?> int)
                elif typeof<'T>.Equals(typeof<byte>)   then sprintf "%3d"  (v |> box :?> byte)
                elif typeof<'T>.Equals(typeof<bool>)   then if (v |> box :?> bool) then "true " else "false"
                else sprintf "%A;" v
            | 1 -> "[" + (String.concat " " (subStrs ())) + "]"
            | _ -> "[" + (String.concat ("\n" + lineSpace) (subStrs ())) + "]"

        prettyDim " " a                       

    type ArrayNDT<'T> with
        /// pretty contents string
        member this.Pretty = pretty 10 this

        /// full contents string
        member this.Full = pretty 0 this


[<AutoOpen>]
module ArrayNDTypes2 =
    type ArrayNDT<'T> = ArrayND.ArrayNDT<'T>


