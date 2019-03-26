namespace Tensor.Expr

open Tensor
open DeepNet.Utils


// TODO: move to Tensor along with CUDA implementation.


/// extrapolation behaviour
[<RequireQualifiedAccess>]
type OutsideInterpolatorRange =
    /// zero outside interpolation range
    | Zero
    /// clamp to nearest value outside interpolation range
    | Nearest

/// interpolation mode
[<RequireQualifiedAccess>]
type InterpolationMode =
    /// linear interpolation
    | Linear
    /// interpolate to the table element left of the argument
    | ToLeft

/// one dimensional linear interpoator
type Interpolator = 
    {
        /// interpolation table
        Table:      OrdRef<ITensor>
        /// minimum argument value
        MinArg:     float list
        /// maximum argument value
        MaxArg:     float list
        /// resolution
        Resolution: float list
        /// interpolation behaviour
        Mode:       InterpolationMode
        /// extrapolation behaviour
        Outside:    OutsideInterpolatorRange list
        /// interpolator for derivative
        Derivative: Interpolator list option
    }        
        
    /// number of dimensions 
    member this.NDims = List.length this.Resolution


/// linear interpolator functions
module Interpolator =

    /// interpolator tables
    //let private tables = new Dictionary<Interpolator, ITensor>()

    /// Cache of numerically calculated derivatives.
    let private cachedDerivs = ConditionalWeakTable<Interpolator, Interpolator list> ()

    /// Creates an n-dimensional linear interpolator,
    /// where table contains the equally spaced function values between minArg and maxArg.
    /// Optionally, an interpolator for the derivative can be specified.
    let create (tbl: ITensor) (minArg: float list) (maxArg: float list) 
               (outside: OutsideInterpolatorRange list) (mode: InterpolationMode) 
               (derivative: Interpolator list option) =
        
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
            Table = OrdRef tbl
            MinArg = minArg
            MaxArg = maxArg
            Resolution = List.zip3 minArg maxArg tbl.Shape
                         |> List.map (fun (min, max, nElems) -> (max - min) / float (nElems - 1L))
            Mode = mode
            Outside = outside
            Derivative = derivative
        }

        ip

    /// Computes the derivative interpolators for the specified interpolator.
    let private computeDerivs (ip: Interpolator) =
        let tbl = ip.Table.Value    

        // Compute derivative for each dimension.
        [0 .. ip.NDims-1]
        |> List.map (fun dd -> 
            let diffTbl =
                match ip.Mode with
                | InterpolationMode.Linear ->
                    let diffFac = ITensor.scalar tbl.Dev (convTo tbl.DataType ip.Resolution.[dd]) 
                    let diffTbl = (ITensor.diffAxis dd tbl).Divide diffFac                                            
                    let zeroShp =
                        [for d, s in List.indexed tbl.Shape do
                            if d = dd then yield 1L
                            else yield s]
                    let zero = ITensor.zeros diffTbl.DataType diffTbl.Dev zeroShp 
                    ITensor.concat dd [diffTbl; zero]
                | InterpolationMode.ToLeft ->
                    ITensor.zeros tbl.DataType tbl.Dev tbl.Shape

            let outside =
                List.indexed ip.Outside
                |> List.map (fun (d, o) -> if d = dd then OutsideInterpolatorRange.Zero else o)
            create diffTbl ip.MinArg ip.MaxArg outside InterpolationMode.ToLeft None)
            
    /// Gets the interpolator for the derivative of the specified one-dimensional interpolator.
    /// If no derivative was specified at creation of the interpolator, it is calculated numerically.
    let getDeriv (derivDim: int) (ip: Interpolator) =
        if not (0 <= derivDim && derivDim < ip.NDims) then
            invalidArg "derivDim" "derivative dimension out of range"

        let ipds = 
            match ip.Derivative with
            | Some ipds -> ipds  // use provided derivative tables
            | None ->            // create derivative table by numeric differentiation
                cachedDerivs.GetValue(ip, fun _ -> computeDerivs ip)              

        ipds.[derivDim]         

    /// Performs interpolation on host.
    let interpolate (ip: Interpolator) (es: Tensor<'T> list) : Tensor<'T> =
        let tbl = ip.Table.Value :?> Tensor<'T>

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
                | _, OutsideInterpolatorRange.Nearest, _ when idx < 0L                  -> interpolateInDim (leftIdxs @ [0L]) x
                | _, OutsideInterpolatorRange.Zero,    _ when idx < 0L                  -> 0.0
                | _, OutsideInterpolatorRange.Nearest, _ when idx > tbl.Shape.[d] - 2L  -> interpolateInDim (leftIdxs @ [tbl.Shape.[d] - 1L]) x
                | _, OutsideInterpolatorRange.Zero,    _ when idx > tbl.Shape.[d] - 2L  -> 0.0
                | _, _, InterpolationMode.Linear -> 
                    let left = interpolateInDim (leftIdxs @ [idx]) x
                    let right = interpolateInDim (leftIdxs @ [idx+1L]) x
                    (1.0 - fac) * left + fac * right
                | _, _, InterpolationMode.ToLeft -> 
                    interpolateInDim (leftIdxs @ [idx]) x

        let res = Tensor.zerosLike es.Head
        for idx in Tensor.allIdx res do
            let x = es |> List.map (fun src -> conv<float> src.[idx])
            res.[idx] <- interpolateInDim [] x |> conv<'T>
        res    


type internal IInterpolator =
    abstract Interpolate: Interpolator -> ITensor list -> ITensor
type TInterpolator<'T> () =
    interface IInterpolator with
        member __.Interpolate ip es = 
            let es = es |> List.map (fun e -> e :?> Tensor<'T>)
            Interpolator.interpolate ip es :> ITensor

type Interpolator with
    /// Performs interpolation on host.
    static member interpolateUntyped (ip: Interpolator) (es: ITensor list) : ITensor =
        (Generic<TInterpolator<_>, IInterpolator> [es.Head.DataType]).Interpolate ip es
