namespace ArrayNDNS

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

    /// ArrayND of any type
    type IArrayNDT =
        abstract Layout:            ArrayNDLayoutT
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

    /// an N-dimensional array with reshape and subview abilities
    [<AbstractClass>]
    [<StructuredFormatDisplay("{PrettyString}")>]
    type ArrayNDT<'T> (layout: ArrayNDLayoutT) =
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
            // slow elementwise fallback copy
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

        abstract Map2Impl: ('T -> 'T -> 'R) -> ArrayNDT<'T> -> ArrayNDT<'R> -> unit
        default this.Map2Impl f other result =
            // slow fallback mapping
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

        member internal this.ToRng (allArgs: obj []) =
            let rec toRng (args: obj list) =
                match args with
                // direct range specification
                | [:? (RangeT list) as rngs] -> rngs

                // slices
                | (:? (int option) as so) :: (:? (int option) as fo)  :: rest ->
                    Rng (so, fo) :: toRng rest
                //  Rng (Some so.Value, Some fo.Value) :: toRng rest
                //| (:? (int option) as so) :: null                     :: rest ->
                //    Rng (Some so.Value, None) :: toRng rest
                //| null                    :: (:? (int option) as fo)  :: rest ->
                //    Rng (None, Some fo.Value) :: toRng rest
                //| null                    :: null                     :: rest ->            
                //    Rng (None, None) :: toRng rest

                // items
                | (:? int as i)           :: rest ->
                    RngElem i :: toRng rest
                | (:? SpecialAxisT as sa) :: rest ->
                    match sa with
                    | NewAxis -> RngNewAxis :: toRng rest
                    | Fill    -> RngAllFill :: toRng rest

                | [] -> []
                | _  -> failwithf "invalid item/slice specification: %A" allArgs 

            allArgs 
            |> Array.toList
            |> toRng

        member this.GetSlice ([<System.ParamArray>] allArgs: obj []) =
            this.View (this.ToRng allArgs) 

        member this.SetSlice ([<System.ParamArray>] allArgs: obj []) =
            let rngArgs = allArgs.[0 .. allArgs.Length - 2] 
            let trgt = this.View (this.ToRng rngArgs) 
            let valueObj = Array.last allArgs
            match valueObj with
            | :? ArrayNDT<'T> as value -> value.CopyTo trgt
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

    /// sequence of all indices 
    let inline allIdx a = layout a |> ArrayNDLayout.allIdx

    /// all indices of the given dimension
    let inline allIdxOfDim dim a = layout a |> ArrayNDLayout.allIdxOfDim dim 
            
    /// sequence of all elements of a ArrayND
    let inline allElems a = allIdx a |> Seq.map (fun i -> get i a)

    /// true if the ArrayND is continguous
    let inline isC a = layout a |> ArrayNDLayout.isC

    /// true if the ArrayND is in Fortran order
    let inline isF a = layout a |> ArrayNDLayout.isF

    /// true if the memory of the ArrayND is a contiguous block
    let inline hasContiguousMemory a = layout a |> ArrayNDLayout.hasContiguousMemory

    /// true if ArrayND can be target of a BLAS operation
    let inline isBlasTargetable a =
        (nDims a = 2) && (isF a)

    /// true if a and b have at least one element in common
    let inline overlapping a b = 
        false // TODO

    /// creates a new ArrayND with the same type as passed and contiguous (row-major) layout for specified shape
    let inline newCOfSameType shp (a: 'A when 'A :> IArrayNDT) : 'A =
        a.NewOfSameType (ArrayNDLayout.newC shp) :?> 'A

    /// creates a new ArrayND with the specified type and contiguous (row-major) layout for specified shape
    let inline newCOfType shp (a: 'A when 'A :> ArrayNDT<_>) =
        a.NewOfType (ArrayNDLayout.newC shp) 

    /// creates a new ArrayND with the same type as passed and Fortran (column-major) layout for specified shape
    let inline newFOfSameType shp (a: 'A when 'A :> ArrayNDT<_>) : 'A =
        a.NewOfSameType (ArrayNDLayout.newF shp) :?> 'A

    /// creates a new ArrayND with the specified type and contiguous (column-major) layout for specified shape
    let inline newFOfType shp (a: 'A when 'A :> ArrayNDT<_>) =
        a.NewOfType (ArrayNDLayout.newF shp) 

    /// creates a new ArrayND with existing data but new layout
    let inline relayout newLayout (a: 'A when 'A :> ArrayNDT<'T>)  =
        a.NewView newLayout :?> 'A

    /// checks that two ArrayNDs have the same shape
    let inline checkSameShape (a: ArrayNDT<'T>) b =
        ArrayNDT<'T>.CheckSameShape a b

    /// Copies all elements from source to destination.
    /// Both ArrayNDs must have the same shape.
    let inline copyTo (source: #ArrayNDT<'T>) (dest: #ArrayNDT<'T>) =
        source.CopyTo dest

    /// Returns a continguous copy of the given ArrayND.
    let inline copy source =
        let dest = newCOfSameType (shape source) source
        copyTo source dest
        dest

    /// Returns a continguous copy of the given IArrayNDT.
    let inline copyUntyped (source: 'T when 'T :> IArrayNDT) =
        source.Copy() :?> 'T

    /// If the ArrayND is not continguous, returns a continguous copy; otherwise
    /// the given ArrayND is returned unchanged.
    let inline ensureC a =
        if isC a then a else copy a

    /// makes a contiguous copy of ary if it is not contiguous and with zero offset
    let inline ensureCAndOffsetFree a = 
        if isC a && offset a = 0 then a else copy a 

    /// inserts a broadcastable dimension of size one as first dimension
    let inline padLeft a =
        relayout (ArrayNDLayout.padLeft (layout a)) a

    /// appends a broadcastable dimension of size one as last dimension
    let inline padRight a =
        relayout (ArrayNDLayout.padRight (layout a)) a

    /// broadcast the given dimensionto the given size
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

    /// swaps the given dimensions
    let inline swapDim ax1 ax2 a =
        relayout (ArrayNDLayout.swapDim ax1 ax2 (layout a)) a

    /// transposes the given matrix
    let inline transpose a =
        relayout (ArrayNDLayout.transpose (layout a)) a

    /// reorders the axes as specified
    let inline reorderAxes (newOrder: int list) a =
        relayout (ArrayNDLayout.reorderAxes newOrder (layout a)) a

    /// creates a subview of an ArrayND
    let inline view ranges a =
        relayout (ArrayNDLayout.view ranges (layout a)) a        

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // array creation functions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// creates a scalar ArrayND of given value and type
    let scalarOfType (value: 'T) (a: 'B when 'B :> ArrayNDT<'T>) : 'B =
        let ary = newCOfSameType [] a
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

    /// fills the specified ArrayND with ones
    let inline fillWithOnes (a: #ArrayNDT<'T>) =
        for idx in allIdx a do
            set idx (ArrayNDT<'T>.One) a

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

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // elementwise operations
    ////////////////////////////////////////////////////////////////////////////////////////////////   
   
    /// Applies the given function elementwise to the given ArrayND and 
    /// stores the result in a new ArrayND.
    let inline map (f: 'T -> 'T) (a: 'A when 'A :> ArrayNDT<'T>) =
        a.Map f :?> 'A

    /// Applies the given function elementwise to the given ArrayND and 
    /// stores the result in a new ArrayND.
    let inline mapTC (f: 'T -> 'R) (a: #ArrayNDT<'T>) =
        a.Map f

    /// Applies the given function elementwise to the given ArrayND inplace.
    let inline mapInplace f (a: #ArrayNDT<'T>) =
        for idx in allIdx a do
            set idx (f (get idx a)) a
            
    /// Applies the given binary function elementwise to the two given ArrayNDs 
    /// and stores the result in a new ArrayND.
    let inline map2 f (a: 'A when 'A :> ArrayNDT<'T>) (b: 'A) =
        a.Map2 f b :?> 'A

    /// Applies the given binary function elementwise to the two given ArrayNDs 
    /// and stores the result in a new ArrayND.
    let inline map2TC (f: 'T -> 'T -> 'R) (a: #ArrayNDT<'T>) (b: #ArrayNDT<'T>) =
        a.Map2 f b 

    /// Applies the given binary function elementwise to the two given ArrayNDs 
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

        // elementwise unary
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

        // elementwise unary logic
        static member (~~~~)    (a: #ArrayNDT<bool>) = map not a

        // elementwise binary
        static member (+) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (+) (+) (+) (+) a b
        static member (-) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (-) (-) (-) (-) a b
        static member (*) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (*) (*) (*) (*) a b
        static member (/) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (/) (/) (/) (/) a b
        static member (%) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) (%) (%) (%) (%) a b
        static member Pow (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2 (unsp) ( ** ) ( ** ) (unsp) (unsp) a b

        // elementwise binary logic
        static member (&&&&) (a: #ArrayNDT<bool>, b: #ArrayNDT<bool>) = map2 (&&) a b
        static member (||||) (a: #ArrayNDT<bool>, b: #ArrayNDT<bool>) = map2 (||) a b

        // elementwise binary comparision
        static member (====) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (=) (=) (=) (=) (=) a b
        static member (<<<<) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (<) (<) (<) (<) (<) a b
        static member (>>>>) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (>) (>) (>) (>) (>) a b
        static member (<<>>) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedMap2TypeChange (<>) (<>) (<>) (<>) (<>) a b

        // elementwise binary with scalars
        static member inline (+) (a: #ArrayNDT<'T>, b: 'T) = a + (scalarOfType b a)
        static member inline (-) (a: #ArrayNDT<'T>, b: 'T) = a - (scalarOfType b a)
        static member inline (*) (a: #ArrayNDT<'T>, b: 'T) = a * (scalarOfType b a)
        static member inline (/) (a: #ArrayNDT<'T>, b: 'T) = a / (scalarOfType b a)
        static member inline (%) (a: #ArrayNDT<'T>, b: 'T) = a % (scalarOfType b a)
        static member inline Pow (a: #ArrayNDT<'T>, b: 'T) = a ** (scalarOfType b a)       
        static member inline (&&&&) (a: #ArrayNDT<bool>, b: bool) = a &&&& (scalarOfType b a)
        static member inline (||||) (a: #ArrayNDT<bool>, b: bool) = a |||| (scalarOfType b a)

        static member inline (+) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfType a b) + b
        static member inline (-) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfType a b) - b
        static member inline (*) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfType a b) * b
        static member inline (/) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfType a b) / b
        static member inline (%) (a: 'T, b: #ArrayNDT<'T>) = (scalarOfType a b) % b
        static member inline Pow (a: 'T, b: #ArrayNDT<'T>) = (scalarOfType a b) ** b
        static member inline (&&&&) (a: bool, b: #ArrayNDT<bool>) = (scalarOfType a b) &&&& b
        static member inline (||||) (a: bool, b: #ArrayNDT<bool>) = (scalarOfType a b) |||| b

        // transposition
        member this.T = transpose this

    /// sign keeping type
    let inline signt (a: #ArrayNDT<'T>) =
        ArrayNDT<'T>.SignT a 

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // reduction operations
    ////////////////////////////////////////////////////////////////////////////////////////////////         

    /// value of scalar array
    let inline value a =
        match nDims a with
        | 0 -> get [] a
        | _ -> failwithf "array of shape %A is not a scalar" (shape a)
      
    /// applies the given reduction function over the given dimension
    let inline axisReduce f dim a =
        let c = newCOfSameType (List.without dim (shape a)) a
        for srcRng, dstIdx in ArrayNDLayout.allSourceRangesAndTargetIdxsForAxisReduction dim (layout a) do
            set dstIdx (f (view srcRng a) |> get []) c
        c

    /// elementwise sum
    let inline sumImpl (a: ArrayNDT<'T>) =
        let value = allElems a |> Seq.fold (+) ArrayNDT<'T>.Zero         
        scalarOfType value a

    /// elementwise sum
    let inline sum (a: #ArrayNDT<'T>) =
        typedApply (unsp) sumImpl sumImpl sumImpl sumImpl a 

    /// elementwise sum over given axis
    let inline sumAxis dim a = axisReduce sum dim a
    
    /// elementwise product
    let inline productImpl (a: ArrayNDT<'T>) =
        let value = allElems a |> Seq.fold (*) ArrayNDT<'T>.One
        scalarOfType value a

    /// elementwise product
    let inline product (a: #ArrayNDT<'T>) =
        typedApply (unsp) productImpl productImpl productImpl productImpl a 

    /// elementwise product over given axis
    let inline productAxis dim a = axisReduce product dim a

    let inline maxImpl a =
        let value = allElems a |> Seq.reduce max
        scalarOfType value a

    let inline max a =
        if nElems a = 0 then invalidArg "a" "cannot compute max of empty ArrayNDT"
        typedApply (unsp) maxImpl maxImpl maxImpl maxImpl a

    let inline minImpl a =
        let value = allElems a |> Seq.reduce min
        scalarOfType value a

    let inline min a =
        if nElems a = 0 then invalidArg "a" "cannot compute min of empty ArrayNDT"
        typedApply (unsp) minImpl minImpl minImpl minImpl a

    let inline all a =
        let value = allElems a |> Seq.fold (&&) true
        scalarOfType value a

    let inline any a =
        let value = allElems a |> Seq.fold (||) false
        scalarOfType value a

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // tensor operations
    ////////////////////////////////////////////////////////////////////////////////////////////////         

    /// dot product implementation between vec*vec, mat*vec, mat*mat
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

        match nDims a, nDims b with
            | 1, 1 when shape a = shape b -> 
                map2 (*) a b |> sum
            | 2, 1 when (shape a).[1] = (shape b).[0] -> 
                matrixDot a (padRight b) |> view [RngAll; RngElem 0] 
            | 2, 2 when (shape a).[1] = (shape b).[0] ->
                matrixDot a b
            | _ -> 
                failwithf "cannot compute dot product between arrays of shapes %A and %A" 
                    (shape a) (shape b)

    type ArrayNDT<'T> with   
        /// dot product
        static member (.*) (a: #ArrayNDT<'T>, b: #ArrayNDT<'T>) = typedApply2 (unsp) dotImpl dotImpl dotImpl dotImpl a b

    /// dot product between vec*vec, mat*vec, mat*mat
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

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // pretty printing
    ////////////////////////////////////////////////////////////////////////////////////////////////         
    
    let prettyString (a: ArrayNDT<'T>) =
        let maxElems = 10

        let rec prettyDim lineSpace a =
            let ls () = (shape a).[0]
            let subPrint idxes = 
                idxes
                |> Seq.map (fun i -> 
                    prettyDim (lineSpace + " ") (view [RngElem i; RngAllFill] a)) 
                |> Seq.toList                   
            let subStrs () = 
                if ls() < maxElems then
                    subPrint (seq {0 .. ls() - 1})
                else
                    let leftIdx = seq {0 .. (maxElems / 2)}
                    let rightIdx = seq {ls() - 1 - (maxElems / 2) + 2 .. (ls() - 1)}
                    (subPrint leftIdx) @ ["..."] @ (subPrint rightIdx)

            match nDims a with
            | 0 -> 
                let v = value a
                if   typeof<'T>.Equals(typeof<single>) then sprintf "%.4f" (v |> box :?> single)
                elif typeof<'T>.Equals(typeof<double>) then sprintf "%.4f" (v |> box :?> double)
                elif typeof<'T>.Equals(typeof<int>)    then sprintf "%4d"  (v |> box :?> int)
                elif typeof<'T>.Equals(typeof<byte>)   then sprintf "%3d"  (v |> box :?> byte)
                else sprintf "%A;" v
            | 1 -> "[" + (String.concat " " (subStrs ())) + "]"
            | _ -> "[" + (String.concat ("\n" + lineSpace) (subStrs ())) + "]"

        prettyDim " " a                       

    type ArrayNDT<'T> with
        /// pretty contents string
        member this.PrettyString = prettyString this


[<AutoOpen>]
module ArrayNDTypes2 =
    type ArrayNDT<'T> = ArrayND.ArrayNDT<'T>


