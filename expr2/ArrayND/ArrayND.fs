namespace ArrayNDNS

open Util


[<AutoOpen>]
module ArrayNDLayoutTypes =
    // layout (shape, offset, stride) of an ArrayND
    type ArrayNDLayoutT =
        {/// shape
         Shape: int list;
         /// offset in elements
         Offset: int;
         /// stride in elements
         Stride: int list;}

    /// range specification
    type RangeT = 
        | Elem of int
        | Rng of int * int
        | NewAxis
        | All


module ArrayNDLayout =
    /// checks that the given index is valid for the given shape
    let inline checkIndex shp idx =
        if List.length shp <> List.length idx then
            failwithf "index %A has other dimensionality than shape %A" idx shp
        if not (List.forall2 (fun s i -> 0 <= i && i < s) shp idx) then 
            failwithf "index %A out of range for shape %A" idx shp

    /// address of element
    let inline addr idx a =
        checkIndex a.Shape idx
        Seq.map2 (*) idx a.Stride |> Seq.fold (+) a.Offset
  
    /// shape 
    let inline shape a = a.Shape

    /// stride
    let inline stride a = a.Stride

    /// offset 
    let inline offset a = a.Offset

    /// number of dimensions
    let inline nDims a = List.length (shape a)

    /// number of elements 
    let inline nElems a = List.fold (*) 1 (shape a)

    /// sequence of all indices 
    let inline allIdx a =
        let rec generate shp = seq {
            match shp with
            | l::ls ->
                for i=0 to l - 1 do
                    for is in generate ls do
                        yield i::is
            | [] -> yield []
        } 
        generate (shape a)

    /// all indices of the given dimension
    let inline allIdxOfDim dim a =
        { 0 .. a.Shape.[dim] - 1}

    /// computes the stride given the shape for the ArrayND to be continguous (row-major)
    let rec contiguousStride (shape: int list) =
        match shape with
        | [] -> []
        | [l] -> [1]
        | l::(lp::lrest) ->
            match contiguousStride (lp::lrest) with 
            | sp::srest -> (lp*sp)::sp::srest
            | [] -> failwith "unexpected"    

    /// computes the stride given the shape for the ArrayND to be in Fortran order (column-major)
    let inline columnMajorStride (shape: int list) =
        let rec buildStride elemsLeft shape =
            match shape with
            | [] -> []
            | l :: ls ->
                elemsLeft :: buildStride (l * elemsLeft) ls
        buildStride 1 shape

    /// a contiguous (row-major) ArrayND layout of the given shape 
    let inline newContiguous shp =
        {Shape=shp; Stride=contiguousStride shp; Offset=0;}

    /// a Fortran (column-major) ArrayND layout of the given shape 
    let inline newColumnMajor shp =
        {Shape=shp; Stride=columnMajorStride shp; Offset=0;}

    /// true if the ArrayND is contiguous
    let inline isContiguous a = (stride a = contiguousStride (shape a))

    /// true if the ArrayND is in Fortran order
    let inline isColumnMajor a = (stride a = columnMajorStride (shape a))

    /// adds a new dimension of size one to the left
    let inline padLeft a =
        {a with Shape=1::a.Shape; Stride=0::a.Stride}

    /// adds a new dimension of size one to the right
    let inline padRight a =
        {a with Shape=a.Shape @ [1]; Stride=a.Stride @ [0]}

    /// broadcast the given dimensionto the given size
    let inline broadcastDim dim size a =
        if size < 0 then invalidArg "size" "size must be positive"
        match (shape a).[dim] with
        | 1 -> {a with Shape=List.set dim size a.Shape; Stride=List.set dim 0 a.Stride}
        | _ -> failwithf "dimension %d of shape %A must be of size 1 to broadcast" dim (shape a)

    /// pads shapes from the left until they have same rank
    let rec padToSame a b =
        if nDims a < nDims b then padToSame (padLeft a) b
        elif nDims b < nDims a then padToSame a (padLeft b)
        else a, b

    /// broadcasts to have the same size
    let inline broadcastToSame ain bin =
        let mutable a, b = padToSame ain bin
        for d = 0 to (nDims a) - 1 do
            match (shape a).[d], (shape b).[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim d bl a
            | al, bl when bl = 1 -> b <- broadcastDim d al b
            | _ -> failwithf "cannot broadcast shapes %A and %A to same size" (shape ain) (shape bin)
        a, b

    /// broadcasts a ArrayND to the given shape
    let inline broadcastToShape bs ain =
        let bsDim = List.length bs
        if bsDim <> nDims ain then
            failwithf "shape %A has different rank than shape %A" bs (shape ain)

        let mutable a = ain
        for d = 0 to bsDim - 1 do
            match (shape a).[d], bs.[d] with
            | al, bl when al = bl -> ()
            | al, bl when al = 1 -> a <- broadcastDim d bl a
            | _ -> failwithf "cannot broadcast shape %A to shape %A" (shape ain) bs
        a

    /// Reshape layout under the assumption that it is contiguous.
    /// The number of elements must not change.
    let inline reshape shp a =
        if not (isContiguous a) then
            invalidArg "a" "layout must be contiguous for reshape"
        let shpElems = List.fold (*) 1 shp
        if shpElems <> nElems a then
            failwithf "cannot reshape from shape %A (with %d elements) to shape %A (with %d elements)" 
                (shape a) (nElems a) shp shpElems
        {a with Shape=shp; Stride=contiguousStride shp;}

    /// swaps the given dimensions
    let inline swapDim ax1 ax2 a =
        let nElems = nElems a
        if not (0 <= ax1 && ax1 < nElems && 0 <= ax2 && ax2 < nElems) then
            failwithf "cannot swap dimension %d with %d of for shape %A" ax1 ax2 (shape a)
        let shp, str = shape a, stride a
        {a with Shape=shp |> List.set ax1 shp.[ax2] |> List.set ax2 shp.[ax1]; 
                Stride=str |> List.set ax1 str.[ax2] |> List.set ax2 str.[ax1];}

    /// transposes the given layout of a matrix
    let inline transpose a =
        if nDims a <> 2 then failwithf "cannot transpose non-matrix of shape %A" (shape a)
        swapDim 0 1 a

    /// creates a subview layout
    let rec view ranges a =
        let checkElementRange isEnd nElems i =
            let nElems = if isEnd then nElems + 1 else nElems
            if not (0 <= i && i < nElems) then
                failwithf "index %d out of range in slice %A for shape %A" i ranges (shape a)
        let failIncompatible () =
            failwithf "slice %A is incompatible with shape %A" ranges (shape a)

        let rec recView ranges a =
            match ranges, a.Shape, a.Stride with
            | (All | Elem _ | Rng _ as idx)::rSlices, shp::rShps, str::rStrs ->
                let ra = recView rSlices {a with Shape=rShps; Stride=rStrs} 
                match idx with 
                | All ->
                    {ra with Shape = shp::ra.Shape;
                             Stride = str::ra.Stride}
                | Elem i -> 
                    checkElementRange false shp i
                    {ra with Offset = ra.Offset + i*str;
                             Stride = ra.Stride;
                             Shape = ra.Shape} 
                | Rng(start, stop) ->
                    checkElementRange false shp start
                    checkElementRange true shp stop
                    {ra with Offset = ra.Offset + start*str;
                             Shape = (stop - start)::ra.Shape;
                             Stride = str::ra.Stride} 
                | NewAxis -> failwith "impossible"
            | NewAxis::rSlices, _, _ ->
                let ra = recView rSlices a
                {ra with Shape = 1::ra.Shape; 
                         Stride = 0::ra.Stride}
            | [], [], _ -> a 
            | _ -> failIncompatible ()         

        recView ranges a

    let allSourceRangesAndTargetIdxsForAxisReduction dim a =
        if not (0 <= dim && dim < nDims a) then
            failwithf "reduction dimension %d out of range for shape %A" dim (shape a)

        let rec generate shape dim = seq {
            match shape with
            | l::ls ->
                let rest = generate ls (dim-1)
                if dim = 0 then
                    for is, ws in rest do
                        yield All::is, ws
                else
                    for i=0 to l - 1 do
                        for is, ws in rest do
                            yield Elem i::is, i::ws
            | [] -> yield [], []
        } 
        generate (shape a) dim  


//[<AutoOpen>]
module ArrayND =

    /// object that can be queried for shape
    type IHasLayout =
        /// shape of object
        abstract member Layout : ArrayNDLayoutT

    /// an N-dimensional array with reshape and subview abilities
    [<AbstractClass>]
    type ArrayNDT<'T> (layout: ArrayNDLayoutT) =
        /// layout
        member this.Layout = layout

        /// value zero (if defined for 'T)
        member inline this.Zero =
            if typeof<'T>.Equals(typeof<double>) then (box 0.0) :?> 'T
            elif typeof<'T>.Equals(typeof<single>) then (box 0.0f) :?> 'T
            elif typeof<'T>.Equals(typeof<int>) then (box 0) :?> 'T
            elif typeof<'T>.Equals(typeof<byte>) then (box 0) :?> 'T
            else failwithf "zero is undefined for type %A" typeof<'T>

        /// value one (if defined for 'T)
        member inline this.One =
            if typeof<'T>.Equals(typeof<double>) then (box 1.0) :?> 'T
            elif typeof<'T>.Equals(typeof<single>) then (box 1.0f) :?> 'T
            elif typeof<'T>.Equals(typeof<int>) then (box 1) :?> 'T
            elif typeof<'T>.Equals(typeof<byte>) then (box 1) :?> 'T
            else failwithf "one is undefined for type %A" typeof<'T>

        /// item access
        abstract Item : int list -> 'T with get, set

        /// a new ArrayND of same type and new storage allocation for given layout
        abstract NewOfSameType : ArrayNDLayoutT -> ArrayNDT<'T>

        /// a new ArrayND of same type with same storage allocation but new layout
        abstract NewView : ArrayNDLayoutT -> ArrayNDT<'T>

        interface IHasLayout with
            member this.Layout = this.Layout

        /// unchecked cast to NDArrayT<'A>
        member this.Cast<'A> () =
            let thisBoxed = box this
            let thisCasted = unbox<ArrayNDT<'A>> thisBoxed
            thisCasted

        /// unchecked cast of v to NDArrayT<'T> (this type)
        member this.CastToMe (v: ArrayNDT<'A>) = v.Cast<'T> ()


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // element access
    ////////////////////////////////////////////////////////////////////////////////////////////////   
    
    /// get element value
    let inline get idx (a: ArrayNDT<_>) = a.[idx]
    
    /// set element value
    let inline set idx value (a: ArrayNDT<_>) = a.[idx] <- value

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

    /// layout
    let inline layout (a: IHasLayout) = a.Layout

    /// number of dimensions
    let inline nDims a = layout a |> ArrayNDLayout.nDims

    /// number of elements 
    let inline nElems a = layout a |> ArrayNDLayout.nElems
    
    /// shape 
    let inline shape a = layout a |> ArrayNDLayout.shape

    /// stride
    let inline stride a = layout a |> ArrayNDLayout.stride

    /// offset 
    let inline offset a = layout a |> ArrayNDLayout.offset

    /// sequence of all indices 
    let inline allIdx a = layout a |> ArrayNDLayout.allIdx

    /// all indices of the given dimension
    let inline allIdxOfDim dim a = layout a |> ArrayNDLayout.allIdxOfDim dim 
            
    /// sequence of all elements of a ArrayND
    let inline allElems a = allIdx a |> Seq.map (fun i -> get i a)

    /// true if the ArrayND is continguous
    let inline isContiguous a = layout a |> ArrayNDLayout.isContiguous

    /// true if the ArrayND is in Fortran order
    let inline isColumnMajor a = layout a |> ArrayNDLayout.isColumnMajor

    /// creates a new ArrayND with the same type as passed and contiguous (row-major) layout for specified shape
    let inline newContiguousOfType shp (a: ArrayNDT<'T>) =
        a.NewOfSameType (ArrayNDLayout.newContiguous shp)

    /// creates a new ArrayND with the same type as passed and Fortran (column-major) layout for specified shape
    let inline newColumnMajorOfType shp (a: ArrayNDT<'T>) =
        a.NewOfSameType (ArrayNDLayout.newColumnMajor shp)

    /// creates a new ArrayND with existing data but new layout
    let inline relayout newLayout (a: ArrayNDT<'T>) =
        a.NewView newLayout

    /// checks that two ArrayNDs have the same shape
    let inline checkSameShape a b =
        if shape a <> shape b then
            failwithf "ArrayNDs of shapes %A and %A were expected to have same shape" (shape a) (shape b)

    /// Copies all elements from source to destination.
    /// Both ArrayNDs must have the same shape.
    let inline copyTo (source: ArrayNDT<'T>) (dest: ArrayNDT<'T>) =
        checkSameShape source dest
        for idx in allIdx source do
            set idx (get idx source) dest

    /// Returns a continguous copy of the given ArrayND.
    let inline copy source =
        let dest = newContiguousOfType (shape source) source
        copyTo source dest
        dest

    /// If the ArrayND is not continguous, returns a continguous copy; otherwise
    /// the given ArrayND is returned unchanged.
    let inline makeContiguous a =
        if isContiguous a then a else copy a

    let inline padLeft a =
        relayout (ArrayNDLayout.padLeft (layout a)) a

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
    let inline reshape shp a =
        reshapeView shp (makeContiguous a)

    /// swaps the given dimensions
    let inline swapDim ax1 ax2 a =
        relayout (ArrayNDLayout.swapDim ax1 ax2 (layout a)) a

    /// transposes the given matrix
    let inline transpose a =
        relayout (ArrayNDLayout.transpose (layout a)) a

    /// creates a subview of an ArrayND
    let inline view ranges a =
        relayout (ArrayNDLayout.view ranges (layout a)) a        

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // array creation functions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// fills the specified ArrayND with zeros
    let inline fillWithZeros (a: ArrayNDT<'T>) =
        for idx in allIdx a do
            set idx (a.Zero) a
   
    /// ArrayND of specified shape and same type as a filled with zeros.
    let inline zerosOfType shp a =
        newContiguousOfType shp a

    /// ArrayND of same shape filled with zeros.
    let inline zerosLike a =
        newContiguousOfType (shape a) a

    /// fills the specified ArrayND with ones
    let inline fillWithOnes (a: ArrayNDT<'T>) =
        for idx in allIdx a do
            set idx (a.One) a

    /// ArrayND of specified shape and same type as a filled with ones.
    let inline onesOfType shp a =
        let n = newContiguousOfType shp a
        fillWithOnes n
        n        

    /// ArrayND of same shape filled with ones.
    let inline onesLike (a: ArrayNDT<'T>) =
        onesOfType (shape a) a

    /// creates a scalar ArrayND of given value and type
    let inline scalarOfType value a =
        let ary = newContiguousOfType [] a
        set [] value ary
        ary

    /// fills the diagonal of a quadratic matrix with ones
    let inline fillDiagonalWithOnes (a: ArrayNDT<'T>) =
        match shape a with
        | [n; m] when n = m ->
            for i = 0 to n - 1 do
                set [i; i] a.One a
        | _ -> invalidArg "a" "need a quadratic matrix"

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // elementwise operations
    ////////////////////////////////////////////////////////////////////////////////////////////////   
   
    /// Applies the given function elementwise to the given ArrayND and 
    /// stores the result in a new ArrayND.
    let inline map f (a: ArrayNDT<'T>) =
        let c = zerosLike a
        for idx in allIdx a do
            set idx (f (get idx a)) c
        c

    /// Applies the given function elementwise to the given ArrayND inplace.
    let inline mapInplace f (a: ArrayNDT<'T>) =
        for idx in allIdx a do
            set idx (f (get idx a)) a
            
    /// Applies the given binary function elementwise to the two given ArrayNDs 
    /// and stores the result in a new ArrayND.
    let inline map2 f (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        let a, b = broadcastToSame a b
        let c = zerosLike a
        for idx in allIdx a do
            let cv = f (get idx a) (get idx b)
            set idx cv c
        c        

    /// Applies the given binary function elementwise to the two given ArrayNDs 
    /// and stores the result in the first ArrayND.
    let inline map2Inplace f (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        let a, b = broadcastToSame a b
        for idx in allIdx a do
            let cv = f (get idx a) (get idx b)
            set idx cv a

    /// unsupported operation for this type
    let inline unsp (a: 'T) : 'R = 
        failwithf "operation unsupported for type %A" typeof<'T>

    let inline uncheckedApply (f: ArrayNDT<'A> -> ArrayNDT<'A>) (a: ArrayNDT<'T>) =
        let aCast = a.Cast<'A> ()
        let mCast = f aCast
        let m = a.CastToMe mCast
        m

    let inline uncheckedApply2 (f: ArrayNDT<'A> -> ArrayNDT<'A> -> ArrayNDT<'A>) (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        let aCast = a.Cast<'A> ()
        let bCast = b.Cast<'A> ()
        let mCast = f aCast bCast
        let m = a.CastToMe mCast
        m

    let inline uncheckedMap (f: 'A -> 'A) (a: ArrayNDT<'T>) =
        uncheckedApply (map f) a

    let inline uncheckedMap2 (f: 'A -> 'A -> 'A) (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        uncheckedApply2 (map2 f) a b

    let inline typedApply   (fDouble: ArrayNDT<double> -> ArrayNDT<double>) 
                            (fSingle: ArrayNDT<single> -> ArrayNDT<single>)
                            (fInt:    ArrayNDT<int>    -> ArrayNDT<int>)
                            (fByte:   ArrayNDT<byte>   -> ArrayNDT<byte>)
                            (a: ArrayNDT<'T>) =
        if   typeof<'T>.Equals(typeof<double>) then uncheckedApply fDouble a 
        elif typeof<'T>.Equals(typeof<single>) then uncheckedApply fSingle a 
        elif typeof<'T>.Equals(typeof<int>)    then uncheckedApply fInt    a 
        elif typeof<'T>.Equals(typeof<byte>)   then uncheckedApply fByte   a 
        else failwith "unknown type"

    let inline typedApply2  (fDouble: ArrayNDT<double> -> ArrayNDT<double> -> ArrayNDT<double>) 
                            (fSingle: ArrayNDT<single> -> ArrayNDT<single> -> ArrayNDT<single>)
                            (fInt:    ArrayNDT<int>    -> ArrayNDT<int>    -> ArrayNDT<int>)
                            (fByte:   ArrayNDT<byte>   -> ArrayNDT<byte>   -> ArrayNDT<byte>)
                            (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        if   typeof<'T>.Equals(typeof<double>) then uncheckedApply2 fDouble a b
        elif typeof<'T>.Equals(typeof<single>) then uncheckedApply2 fSingle a b
        elif typeof<'T>.Equals(typeof<int>)    then uncheckedApply2 fInt    a b
        elif typeof<'T>.Equals(typeof<byte>)   then uncheckedApply2 fByte   a b
        else failwith "unknown type"

    let inline typedMap (fDouble: double -> double) 
                        (fSingle: single -> single)
                        (fInt:    int    -> int)
                        (fByte:   byte   -> byte)
                        (a: ArrayNDT<'T>) =
        typedApply (map fDouble) (map fSingle) (map fInt) (map fByte) a

    let inline typedMap2 (fDouble: double -> double -> double) 
                         (fSingle: single -> single -> single)
                         (fInt:    int    -> int    -> int)
                         (fByte:   byte   -> byte   -> byte)
                         (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        typedApply2 (map2 fDouble) (map2 fSingle) (map2 fInt) (map2 fByte) a b


    type ArrayNDT<'T> with    

        // elementwise unary
        static member (~-) (a: ArrayNDT<'T>) = typedMap (~-) (~-) (~-) (unsp) a
        static member Log (a: ArrayNDT<'T>) = typedMap log log (unsp) (unsp) a
        static member Exp (a: ArrayNDT<'T>) = typedMap exp exp (unsp) (unsp) a

        // elementwise binary
        static member (+) (a: ArrayNDT<'T>, b: ArrayNDT<'T>) = typedMap2 (+) (+) (+) (+) a b
        static member (-) (a: ArrayNDT<'T>, b: ArrayNDT<'T>) = typedMap2 (-) (-) (-) (-) a b
        static member (*) (a: ArrayNDT<'T>, b: ArrayNDT<'T>) = typedMap2 (*) (*) (*) (*) a b
        static member (/) (a: ArrayNDT<'T>, b: ArrayNDT<'T>) = typedMap2 (/) (/) (/) (/) a b
        static member Pow (a: ArrayNDT<'T>, b: ArrayNDT<'T>) = typedMap2 ( ** ) ( ** ) (unsp) (unsp) a b

        // elementwise binary with scalars
        static member (+) (a: ArrayNDT<'T>, b: 'T) = a + (scalarOfType b a)
        static member (-) (a: ArrayNDT<'T>, b: 'T) = a - (scalarOfType b a)
        static member (*) (a: ArrayNDT<'T>, b: 'T) = a * (scalarOfType b a)
        static member (/) (a: ArrayNDT<'T>, b: 'T) = a / (scalarOfType b a)
        static member Pow (a: ArrayNDT<'T>, b: 'T) = a / (scalarOfType b a)

        static member (+) (a: 'T, b: ArrayNDT<'T>) = (scalarOfType a b) - b
        static member (-) (a: 'T, b: ArrayNDT<'T>) = (scalarOfType a b) - b
        static member (*) (a: 'T, b: ArrayNDT<'T>) = (scalarOfType a b) - b
        static member (/) (a: 'T, b: ArrayNDT<'T>) = (scalarOfType a b) - b
        static member Pow (a: 'T, b: ArrayNDT<'T>) = (scalarOfType a b) - b

        // transposition
        member this.T = transpose this

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
        let c = newContiguousOfType (List.without dim (shape a)) a
        for srcRng, dstIdx in ArrayNDLayout.allSourceRangesAndTargetIdxsForAxisReduction dim (layout a) do
            set dstIdx (f (view srcRng a) |> get []) c
        c

    /// elementwise sum
    let inline sum a =
        let value = allElems a |> Seq.fold (+) a.Zero         
        scalarOfType value a

    /// elementwise sum over given axis
    let inline sumAxis dim a = axisReduce sum dim a
    
    /// elementwise product
    let inline product a =
        let value = allElems a |> Seq.fold (*) a.One
        scalarOfType value a

    /// elementwise product over given axis
    let inline productAxis dim a = axisReduce product dim a

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // tensor operations
    ////////////////////////////////////////////////////////////////////////////////////////////////         

    /// dot product implementation between vec*vec, mat*vec, mat*mat
    let inline dotImpl (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) =
        let inline matrixDot a b =
            let nI = (shape a).[0]
            let nJ = (shape a).[1]
            let nK = (shape b).[1]
            let c = newContiguousOfType [nI; nK] a
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
                matrixDot a (padRight b) |> view [All; Elem 0] 
            | 2, 2 when (shape a).[1] = (shape b).[0] ->
                matrixDot a b
            | _ -> 
                failwithf "cannot compute dot product between arrays of shapes %A and %A" 
                    (shape a) (shape b)

    type ArrayNDT<'T> with   
        /// dot product
        static member (.*) (a: ArrayNDT<'T>, b: ArrayNDT<'T>) = typedApply2 dotImpl dotImpl dotImpl dotImpl a b

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
        let joined = newContiguousOfType joinedShape tmplArray
        let startPos = List.replicate (List.length joinedShape) 0

        for pos, ary in blockPosAndContents 0 startPos bs do
            let slice = List.map2 (fun p s -> Rng(p, p + s)) pos (shape ary)
            let joinedSlice = joined |> view slice 
            copyTo ary joinedSlice

        joined
    
    /// tensor product
    let inline tensorProduct (a: ArrayNDT<'T>) (b: ArrayNDT<'T>) : ArrayNDT<'T> =
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
        /// dot product
        static member (%*) (a: ArrayNDT<'T>, b: ArrayNDT<'T>) = typedApply2 tensorProduct tensorProduct tensorProduct tensorProduct a b
        
