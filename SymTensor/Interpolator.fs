namespace SymTensor

open Basics
open Tensor


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
            Derivative: InterpolatorT list option
        }        
        
        member this.NDims = List.length this.Resolution



/// linear interpolator functions
module Interpolator =

    /// interpolator tables
    let private tables = new Dictionary<InterpolatorT, ITensor>()

    /// numerically calculated derivatives
    let private numDerivatives = new Dictionary<InterpolatorT, InterpolatorT list>()

    /// Creates an n-dimensional linear interpolator,
    /// where table contains the equally spaced function values between minArg and maxArg.
    /// Optionally, an interpolator for the derivative can be specified.
    let create (tbl: Tensor<'T>) (minArg: float list) (maxArg: float list) 
               (outside: OutsideInterpolatorRangeT list) (mode: InterpolationModeT) 
               (derivative: InterpolatorT list option) =
        
        // check arguments
        let nDims = minArg.Length
        if maxArg.Length <> nDims || outside.Length <> nDims then
            failwith "minArg, maxArg and outside have inconsistent lengths"
        if tbl.NDims <> nDims then failwith "interpolation table has wrong number of dimensions"
        if tbl.Shape |> List.exists (fun e -> e < 2L) then failwith "interpolation table must contain at least 2 entries"
        match derivative with
        | Some ds -> 
            if ds.Length <> nDims then
                failwith "must specify one derivative w.r.t. each input dimension"
            ds |> List.iter (fun d -> 
                if d.NDims <> nDims then failwith "derivatives must have same number of dimensions")
        | None -> ()

        // create interpolator
        let ip = {
            Id = tables.Count
            TypeName = TypeName.ofType<'T>
            MinArg = minArg
            MaxArg = maxArg
            Resolution = List.zip3 minArg maxArg tbl.Shape
                         |> List.map (fun (min, max, nElems) -> (max - min) / float (nElems - 1L))
            Mode = mode
            Outside = outside
            Derivative = derivative
        }

        // store interpolation table
        tables.LockedAdd (ip, tbl)
        ip

    /// Gets the function value table for the specified one-dimensional interpolator as an IArrayNDT.
    let getTableAsIArrayNDT ip =
        match tables.LockedTryFind ip with
        | Some ip -> ip
        | None -> failwithf "interpolator %A is unknown" ip

    /// Gets the function value table for the specified one-dimensional interpolator.
    let getTable<'T> ip =
        getTableAsIArrayNDT ip :?> Tensor<'T>

    type private GetDerivative =
        static member Do<'T> (derivDim: int, ip: InterpolatorT) =
            if not (0 <= derivDim && derivDim < ip.NDims) then
                invalidArg "derivDim" "derivative dimension out of range"

            let ipds = 
                match ip.Derivative with
                | Some ipds -> ipds  // use provided derivative tables
                | None ->            // create derivative table by numeric differentiation
                    match numDerivatives.LockedTryFind ip with
                    | Some ipds -> ipds // use cached derivate table
                    | None ->           // build derivative tables
                        let tbl : Tensor<'T> = getTable ip    

                        // hack to work around slow ArrayNDCuda operations
                        let tbl, wasOnDev = 
                            match tbl.Device with
                            | dev when dev=CudaTensor.Dev -> HostTensor.transfer tbl, true
                            | _ -> tbl, false

                        let ipds = 
                            [0 .. ip.NDims-1]
                            |> List.map (fun dd -> 
                                let diffTbl =
                                    match ip.Mode with
                                    | InterpolateLinearaly ->
                                        let diffFac = Tensor.scalarLike tbl (conv<'T> ip.Resolution.[dd]) 
                                        let diffTbl = Tensor.diffAxis dd tbl / diffFac                                            
                                        let zeroShp =
                                            [for d, s in List.indexed tbl.Shape do
                                                if d = dd then yield 1L
                                                else yield s]
                                        let zero = Tensor.zeros diffTbl.Device zeroShp 
                                        Tensor.concat dd [diffTbl; zero]
                                    | InterpolateToLeft ->
                                        Tensor.zerosLike tbl

                                // hack to work around slow ArrayNDCuda operations
                                let diffTbl =
                                    if wasOnDev then CudaTensor.transfer diffTbl
                                    else diffTbl

                                let outside =
                                    List.indexed ip.Outside
                                    |> List.map (fun (d, o) -> if d = dd then Zero else o)
                                create diffTbl ip.MinArg ip.MaxArg outside InterpolateToLeft None
                            )

                        // cache built tables
                        numDerivatives.LockedAdd (ip, ipds)
                        ipds
            ipds.[derivDim]

    /// Gets the interpolator for the derivative of the specified one-dimensional interpolator.
    /// If no derivative was specified at creation of the interpolator, it is calculated numerically.
    let getDerivative (derivDim: int) (ip: InterpolatorT) =
        callGeneric<GetDerivative, InterpolatorT> "Do" [ip.TypeName.Type] (derivDim, ip)                

    /// Performs interpolation on host.
    let interpolate (ip: InterpolatorT) (es: Tensor<'T> list) : Tensor<'T> =
        let tbl : Tensor<'T> = getTable ip

        /// returns interpolation in dimensions to the right of leftIdxs
        let rec interpolateInDim (leftIdxs: int64 list) (x: float list) =
            let d = leftIdxs.Length
            if d = ip.NDims then
                conv<float> tbl.[leftIdxs]
            else 
                let pos = (x.[d] - ip.MinArg.[d]) / ip.Resolution.[d]
                let posLeft = floor pos 
                let fac = pos - posLeft
                let idx = int64 posLeft 
                match idx, ip.Outside.[d], ip.Mode with
                | _, Nearest, _ when idx < 0L                  -> interpolateInDim (leftIdxs @ [0L]) x
                | _, Zero,    _ when idx < 0L                  -> 0.0
                | _, Nearest, _ when idx > tbl.Shape.[d] - 2L  -> interpolateInDim (leftIdxs @ [tbl.Shape.[d] - 1L]) x
                | _, Zero,    _ when idx > tbl.Shape.[d] - 2L  -> 0.0
                | _, _, InterpolateLinearaly -> 
                    let left = interpolateInDim (leftIdxs @ [idx]) x
                    let right = interpolateInDim (leftIdxs @ [idx+1L]) x
                    (1.0 - fac) * left + fac * right
                | _, _, InterpolateToLeft -> 
                    interpolateInDim (leftIdxs @ [idx]) x

        let res = Tensor.zerosLike es.Head
        for idx in Tensor.allIdx res do
            let x = es |> List.map (fun src -> conv<float> src.[idx])
            res.[idx] <- interpolateInDim [] x |> conv<'T>
        res    

