namespace SymTensor

open Basics
open ArrayNDNS


[<AutoOpen>]
/// linear interpolator types
module InterpolatorTypes =

    /// extrapolation behaviour
    type OutsideInterpolatorRangeT =
        /// zero outside interpolation range
        | Zero
        /// clamp to nearest value outside interpolation range
        | Nearest

    /// interpolation mode
    type InterpolationModeT =
        /// linear interpolation
        | InterpolateLinearaly
        /// interpolate to the table element left of the argument
        | InterpolateToLeft


    /// one dimensional linear interpoator
    type InterpolatorT = 
        {
            /// ID
            Id:         int
            /// data type
            TypeName:   TypeNameT
            /// minimum argument value
            MinArg:     float list
            /// maximum argument value
            MaxArg:     float list
            /// resolution
            Resolution: float list
            /// interpolation behaviour
            Mode:       InterpolationModeT
            /// extrapolation behaviour
            Outside:    OutsideInterpolatorRangeT list
            /// interpolator for derivative
            Derivative: InterpolatorT option
        }        
        
        member this.NDims = List.length this.Resolution



/// linear interpolator functions
module Interpolator =

    /// interpolator tables
    let private tablesOfInterpolators = new Dictionary<InterpolatorT, IArrayNDT>()

    /// interpolator derivatives
    let private derivativeOfInterpolators = new Dictionary<InterpolatorT, InterpolatorT>()

    /// Creates an n-dimensional linear interpolator,
    /// where table contains the equally spaced function values between minArg and maxArg.
    /// Optionally, an interpolator for the derivative can be specified.
    let createInterpolator (tbl: ArrayNDT<'T>) (minArg: float list) (maxArg: float list) 
                           (outside: OutsideInterpolatorRangeT list) (mode: InterpolationModeT) 
                           (derivative: InterpolatorT option) =
        let nDims = minArg.Length
        if maxArg.Length <> nDims || outside.Length <> nDims then
            failwith "minArg, maxArg and outside have inconsistent lengths"
        if tbl.NDims <> nDims then failwith "interpolation table has wrong number of dimensions"
        if tbl.Shape |> List.exists (fun e -> e < 2) then failwith "interpolation table must contain at least 2 entries"
        let ip = {
            Id = tablesOfInterpolators.Count
            TypeName = TypeName.ofType<'T>
            MinArg = minArg
            MaxArg = maxArg
            Resolution = List.zip3 minArg maxArg tbl.Shape
                         |> List.map (fun (min, max, nElems) -> (max - min) / float (nElems - 1))
            Mode = mode
            Outside = outside
            Derivative = derivative
        }
        lock tablesOfInterpolators
            (fun () -> tablesOfInterpolators.Add (ip, tbl))        
        ip

    /// Gets the function value table for the specified one-dimensional interpolator as an IArrayNDT.
    let getInterpolatorTableAsIArrayNDT ip =
        match tablesOfInterpolators.TryFind ip with
        | Some ip -> ip
        | None -> failwithf "interpolator %A is unknown" ip

    /// Gets the function value table for the specified one-dimensional interpolator.
    let getInterpolatorTable<'T> ip =
        getInterpolatorTableAsIArrayNDT ip :?> ArrayNDT<'T>

    type private GetDerivativeOfInterpolator =
        static member Do<'T> (derivDim: int, ip: InterpolatorT) =
            if not (0 <= derivDim && derivDim < ip.NDims) then
                invalidArg "derivDim" "derivative dimension out of range"

            match ip.Derivative with
            | Some ipd -> ipd  // use provided derivative table
            | None ->          // create derivative table by numeric differentiation
                match derivativeOfInterpolators.TryFind ip with
                | Some ipd -> ipd
                | None ->
                    let tbl = getInterpolatorTable ip                 
                    let diffTbl =
                        match ip.Mode with
                        | InterpolateLinearaly ->
                            let diffTbl = 
                                ArrayND.diffAxis derivDim tbl / 
                                ArrayND.scalarOfSameType tbl (conv<'T> ip.Resolution.[derivDim]) 
                            let zeroShp =
                                [for d, s in List.indexed tbl.Shape do
                                    if d = derivDim then yield 1
                                    else yield s]
                            let zero = ArrayND.zerosOfSameType zeroShp diffTbl
                            ArrayND.concat derivDim [diffTbl; zero]
                        | InterpolateToLeft ->
                            ArrayND.zerosLike tbl
                    let outside =
                        [for d, o in List.indexed ip.Outside do
                            if d = derivDim then yield Zero
                            else yield o]
                    let ipd = createInterpolator diffTbl ip.MinArg ip.MaxArg outside InterpolateToLeft None
                    lock derivativeOfInterpolators
                        (fun () -> derivativeOfInterpolators.Add (ip, ipd))
                    ipd

    /// Gets the interpolator for the derivative of the specified one-dimensional interpolator.
    /// If no derivative was specified at creation of the interpolator, it is calculated numerically.
    let getDerivativeOfInterpolator (derivDim: int) (ip: InterpolatorT) =
        callGeneric<GetDerivativeOfInterpolator, InterpolatorT> "Do" [ip.TypeName.Type] (derivDim, ip)                

    /// Performs interpolation on host.
    let interpolate (ip: InterpolatorT) (es: ArrayNDHostT<'T> list) : ArrayNDHostT<'T> =
        let tbl : ArrayNDT<'T> = getInterpolatorTable ip

        /// returns interpolation in dimensions to the right of leftIdxs
        let rec interpolateInDim (leftIdxs: int list) (x: float list) =
            let d = leftIdxs.Length
            if d = ip.NDims then
                conv<float> tbl.[leftIdxs]
            else 
                let pos = (x.[d] - ip.MinArg.[d]) / ip.Resolution.[d]
                let posLeft = floor pos 
                let fac = pos - posLeft
                let idx = int posLeft 
                match idx, ip.Outside.[d], ip.Mode with
                | _, Nearest, _ when idx < 0                 -> interpolateInDim (leftIdxs @ [0]) x
                | _, Zero,    _ when idx < 0                 -> 0.0
                | _, Nearest, _ when idx > tbl.Shape.[d] - 2 -> interpolateInDim (leftIdxs @ [tbl.Shape.[d] - 1]) x
                | _, Zero,    _ when idx > tbl.Shape.[d] - 2 -> 0.0
                | _, _, InterpolateLinearaly -> 
                    let left = interpolateInDim (leftIdxs @ [idx]) x
                    let right = interpolateInDim (leftIdxs @ [idx+1]) x
                    (1.0 - fac) * left + fac * right
                | _, _, InterpolateToLeft -> 
                    interpolateInDim (leftIdxs @ [idx]) x

        let res = ArrayND.zerosLike es.Head
        for idx in ArrayND.allIdx res do
            let x = es |> List.map (fun src -> conv<float> src.[idx])
            res.[idx] <- interpolateInDim [] x |> conv<'T>
        res    

