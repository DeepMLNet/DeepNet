namespace GPAct


open ArrayNDNS
open SymTensor
open System.Text.RegularExpressions


[<AutoOpen>]
module Utils =

    ///active patterns
    let (|Cuda|Host|) input = if input = "DevCuda" then Cuda else Host

    let inStringToStringOption string =
        let m = Regex.Match ("Some (\S+)", string)
        if m.Success then
            Some m.Groups.[0].Value
        else None
    
    let inStringToIntOption string =
        let m = Regex.Match ("Some (\d+)", string)
        if m.Success then
            let mutable intvalue = 0
            if System.Int32.TryParse(m.Groups.[0].Value, &intvalue) then Some(intvalue)
            else None
        else None
    
    let inStringToFloatOption string =
        let m = Regex.Match ("Some (\d+)", string)
        if m.Success then
            let mutable floatvalue = 0.0
            if System.Double.TryParse(m.Groups.[0].Value, &floatvalue) then Some(floatvalue)
            else None
        else None 

[<AutoOpen>]
module GPUtilsTypes =

    /// initialization types
    type InitMethod =
        /// constant value
        | Const of value:single
        /// linear spaced
        | Linspaced of first:single * last:single
        /// random
        | Random of lower:single * upper:single
        /// identity matrix
        | IdentityMatrix
        /// fan-in/out optimal random weight matrix for neurons
        | FanOptimal
    


module GPUtils =

    /// calculates initialization values
    let initVals initType seed shp =
        let rng = System.Random seed            
        match initType with
        | Const value -> ArrayNDHost.filled shp value
        | Linspaced (first, last) -> 
            ArrayNDHost.linSpaced first last shp.[1]
            |> ArrayND.padLeft
            |> ArrayND.replicate 0 shp.[0]
        | Random (lower, upper) ->
            rng.UniformArrayND (lower, upper) shp
        | IdentityMatrix ->
            match shp with
            | [n; m] when n = m -> ArrayNDHost.identity n
            | _ -> failwith "need square matrix shape for identity matrix initialization"
        | FanOptimal ->
            let fanOut = shp.[0] |> single
            let fanIn = shp.[1] |> single
            let r = 4.0f * sqrt (6.0f / (fanIn + fanOut))
            rng.UniformArrayND (-r, r) shp

    /// Allows the gradient to pass if trainable is true.
    let gate trainable expr =
        if trainable then expr else Expr.assumeZeroDerivative expr

    /// creates a zero covariance matrix for the given input.
    let covZero input =
        // input [smpl, unit]
        let nSmpls = (Expr.shapeOf input).[0]
        let nInput = (Expr.shapeOf input).[1]
        // [smpl,inp1,1] .* [smpl,1,in2] => [smpl,in1,in2]
        // is equivalent to [smpl,inp1,1*] * [smpl,1*,in2] => [smpl,in1,in2]
        Expr.zeros<single> [nSmpls; nInput; nInput]