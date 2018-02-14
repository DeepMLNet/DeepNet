namespace Tensor.Cuda

open System
open System.IO
open System.Threading
open System.Reflection
open System.Reflection.Emit
open System.Runtime.InteropServices
open System.Security.Cryptography
open System.Text
open System.Text.RegularExpressions

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor.Utils
open KernelHelpers



/// CUDA kernels for the CUDA tensor backend
type internal TensorKernels private (dataType: Type, nDims: int) as this =
    inherit CudaModule()

    static let instances = InstanceCache TensorKernels   
    static let headers = ["Elemwise.cuh"; "Reduction.cuh"]

    /// gets kernels of specifed name and argTypes, when dataType is in supTypes (if not empty)
    /// and not in unsupTypes
    let getKernel name argTypes supTypes unsupTypes =
        let supported =
            match supTypes with
            | [] -> not (unsupTypes |> List.contains dataType) 
            | _ ->
                (supTypes |> List.contains dataType) && 
                    not (unsupTypes |> List.contains dataType) 
        if supported then this.GetKernel name argTypes
        else (fun _ -> 
            sprintf "the operation %s is unsupported for tensor data type %A" name dataType
            |> invalidOp)

    let fullTensor = ArgTypeTensor {DataType=dataType; NDims=nDims}
    let reductionSrcTensor = ArgTypeTensor {DataType=dataType; NDims=nDims+1}        
    let argReductionTrgtTensor = ArgTypeTensor {DataType=typeof<int64>; NDims=nDims-1}        
    let boolTensor = ArgTypeTensor {DataType=typeof<bool>; NDims=nDims}
    let scalar = ArgTypeScalar dataType

    // noary kernels
    let fillConst        = getKernel "FillConst" [scalar; fullTensor] [] []
    let fillIncrementing = getKernel "FillIncrementing" [scalar; scalar; fullTensor] [] []

    // unary kernels
    let getUnaryKernel name = getKernel name [fullTensor; fullTensor] 
    let copy        = getUnaryKernel "Copy" [] []
    let unaryPlus   = getUnaryKernel "UnaryPlus" [] []
    let unaryMinus  = getUnaryKernel "UnaryMinus" [] []
    let abs         = getUnaryKernel "Abs" numTypes []
    let sgn         = getUnaryKernel "Sgn" [] []
    let log         = getUnaryKernel "Log" fpTypes []
    let log10       = getUnaryKernel "Log10" fpTypes [] 
    let exp         = getUnaryKernel "Exp" fpTypes []
    let sin         = getUnaryKernel "Sin" fpTypes [] 
    let cos         = getUnaryKernel "Cos" fpTypes []
    let tan         = getUnaryKernel "Tan" fpTypes []
    let asin        = getUnaryKernel "Asin" fpTypes []
    let acos        = getUnaryKernel "Acos" fpTypes []
    let atan        = getUnaryKernel "Atan" fpTypes []
    let sinh        = getUnaryKernel "Sinh" fpTypes []
    let cosh        = getUnaryKernel "Cosh" fpTypes []
    let tanh        = getUnaryKernel "Tanh" fpTypes []
    let sqrt        = getUnaryKernel "Sqrt" fpTypes []
    let ceiling     = getUnaryKernel "Ceiling" fpTypes []
    let floor       = getUnaryKernel "Floor" fpTypes []
    let round       = getUnaryKernel "Round" fpTypes []
    let truncate    = getUnaryKernel "Truncate" fpTypes []
    let negate      = getUnaryKernel "Negate" boolTypes []
     
    // binary kernels
    let getBinaryKernel name = getKernel name [fullTensor; fullTensor; fullTensor] 
    let add         = getBinaryKernel "Add" [] []
    let subtract    = getBinaryKernel "Subtract" [] []
    let multiply    = getBinaryKernel "Multiply" [] []
    let divide      = getBinaryKernel "Divide" [] []
    let modulo      = getBinaryKernel "Modulo" numTypes []
    let power       = getBinaryKernel "Power" fpTypes []
    let minElemwise = getBinaryKernel "MinElemwise" [] []
    let maxElemwise = getBinaryKernel "MaxElemwise" [] []
    let andFn       = getBinaryKernel "And" boolTypes []
    let orFn        = getBinaryKernel "Or" boolTypes []
    let xorFn       = getBinaryKernel "Xor" boolTypes []

    // comparison kernels
    let getComparisonKernel name = getKernel name [boolTensor; fullTensor; fullTensor] 
    let isFinite        = getKernel "IsFinite" [boolTensor; fullTensor] numTypes []
    let equal           = getComparisonKernel "Equal" [] []
    let notEqual        = getComparisonKernel "NotEqual" [] []
    let less            = getComparisonKernel "Less" [] []
    let lessOrEqual     = getComparisonKernel "LessOrEqual" [] []
    let greater         = getComparisonKernel "Greater" [] []
    let greaterOrEqual  = getComparisonKernel "GreaterOrEqual" [] []

    // conditional if-then-else
    let ifThenElse      = getKernel "IfThenElse" [fullTensor; boolTensor; fullTensor; fullTensor] [] []

    // axis reduce kernels
    let getAxisReduceKernel name = getKernel name [scalar; fullTensor; reductionSrcTensor] 
    let minLastAxis     = getAxisReduceKernel "MinLastAxis" [] []
    let maxLastAxis     = getAxisReduceKernel "MaxLastAxis" [] []
    let sumLastAxis     = getAxisReduceKernel "SumLastAxis" [] []
    let productLastAxis = getAxisReduceKernel "ProductLastAxis" [] []
    let allLastAxis     = getAxisReduceKernel "AllLastAxis" boolTypes []
    let anyLastAxis     = getAxisReduceKernel "AnyLastAxis" boolTypes []

    // axis reduce to index kernels
    let getArgAxisReduceKernel name supTypes unsupTypes = 
        if nDims > 0 then
            getKernel name [scalar; argReductionTrgtTensor; fullTensor] supTypes unsupTypes
        else
            (fun _ -> failwith "ArgAxisReduceKernel requires at least a vector")
    let argMinLastAxis  = getArgAxisReduceKernel "ArgMinLastAxis" [] []
    let argMaxLastAxis  = getArgAxisReduceKernel "ArgMaxLastAxis" [] []
    
    // find element index kernel
    let findLastAxis  = 
        if nDims > 0 then
            getKernel "FindLastAxis" [scalar; argReductionTrgtTensor; fullTensor] [] []
        else
            (fun _ -> failwith "FindAxisKernel requires at least a vector")
        
    do this.Build (headers)

    member this.FillConst (stream, value: obj, trgt: NativeTensor) = 
        fillConst (stream, workDimForElemwise trgt, [|value; box trgt|])

    member this.FillIncrementing (stream, start: obj, incr: obj, trgt: NativeTensor) = 
        fillIncrementing (stream, workDimForElemwise trgt, [|start; incr; box trgt|])        

    member this.Copy (stream, trgt: NativeTensor, src: NativeTensor) = 
        copy (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.UnaryPlus (stream, trgt: NativeTensor, src: NativeTensor) = 
        unaryPlus (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.UnaryMinus (stream, trgt: NativeTensor, src: NativeTensor) = 
        unaryMinus (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Abs (stream, trgt: NativeTensor, src: NativeTensor) = 
        abs (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Sgn (stream, trgt: NativeTensor, src: NativeTensor) = 
        sgn (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Log (stream, trgt: NativeTensor, src: NativeTensor) = 
        log (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Log10 (stream, trgt: NativeTensor, src: NativeTensor) = 
        log10 (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Exp (stream, trgt: NativeTensor, src: NativeTensor) = 
        exp (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Sin (stream, trgt: NativeTensor, src: NativeTensor) = 
        sin (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Cos (stream, trgt: NativeTensor, src: NativeTensor) = 
        cos (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Tan (stream, trgt: NativeTensor, src: NativeTensor) = 
        tan (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Asin (stream, trgt: NativeTensor, src: NativeTensor) = 
        asin (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Acos (stream, trgt: NativeTensor, src: NativeTensor) = 
        acos (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Atan (stream, trgt: NativeTensor, src: NativeTensor) = 
        atan (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Sinh (stream, trgt: NativeTensor, src: NativeTensor) = 
        sinh (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Cosh (stream, trgt: NativeTensor, src: NativeTensor) = 
        cosh (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Tanh (stream, trgt: NativeTensor, src: NativeTensor) = 
        tanh (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Sqrt (stream, trgt: NativeTensor, src: NativeTensor) = 
        sqrt (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Ceiling (stream, trgt: NativeTensor, src: NativeTensor) = 
        ceiling (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Floor (stream, trgt: NativeTensor, src: NativeTensor) = 
        floor (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Round (stream, trgt: NativeTensor, src: NativeTensor) = 
        round (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Truncate (stream, trgt: NativeTensor, src: NativeTensor) = 
        truncate (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Negate (stream, trgt: NativeTensor, src: NativeTensor) = 
        negate (stream, workDimForElemwise trgt, [|box trgt; box src|])

    member this.Add (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        add (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.Subtract (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        subtract (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.Multiply (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        multiply (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.Divide (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        divide (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.Modulo (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        modulo (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.Power (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        power (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.MinElemwise (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        minElemwise (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.MaxElemwise (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        maxElemwise (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.IsFinite (stream, trgt: NativeTensor, src1: NativeTensor) = 
        isFinite (stream, workDimForElemwise trgt, [|box trgt; box src1|])

    member this.Equal (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        equal (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.NotEqual (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        notEqual (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.Less (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        less (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.LessOrEqual (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        lessOrEqual (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.Greater (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        greater (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.GreaterOrEqual (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        greaterOrEqual (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.IfThenElse (stream, trgt: NativeTensor, cond: NativeTensor,
                            ifTrue: NativeTensor, ifFalse: NativeTensor) = 
        ifThenElse (stream, workDimForElemwise trgt, [|box trgt; box cond; box ifTrue; box ifFalse|])    

    member this.And (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        andFn (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.Or (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        orFn (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.Xor (stream, trgt: NativeTensor, src1: NativeTensor, src2: NativeTensor) = 
        xorFn (stream, workDimForElemwise trgt, [|box trgt; box src1; box src2|])

    member this.MinLastAxis (stream, trgt: NativeTensor, src: NativeTensor) =         
        let initial = maxValueOf dataType
        minLastAxis (stream, workDimForElemwise trgt, [|initial; box trgt; box src|])

    member this.MaxLastAxis (stream, trgt: NativeTensor, src: NativeTensor) =         
        let initial = minValueOf dataType
        maxLastAxis (stream, workDimForElemwise trgt, [|initial; box trgt; box src|])

    member this.SumLastAxis (stream, trgt: NativeTensor, src: NativeTensor) =         
        let initial = convTo dataType 0
        sumLastAxis (stream, workDimForElemwise trgt, [|initial; box trgt; box src|])

    member this.ProductLastAxis (stream, trgt: NativeTensor, src: NativeTensor) =         
        let initial = convTo dataType 1
        productLastAxis (stream, workDimForElemwise trgt, [|initial; box trgt; box src|])

    member this.AllLastAxis (stream, trgt: NativeTensor, src: NativeTensor) =         
        let initial = true
        allLastAxis (stream, workDimForElemwise trgt, [|initial; box trgt; box src|])

    member this.AnyLastAxis (stream, trgt: NativeTensor, src: NativeTensor) =         
        let initial = false
        anyLastAxis (stream, workDimForElemwise trgt, [|initial; box trgt; box src|])

    member this.ArgMinLastAxis (stream, trgt: NativeTensor, src: NativeTensor) =         
        let initial = maxValueOf dataType
        argMinLastAxis (stream, workDimForElemwise trgt, [|initial; box trgt; box src|])

    member this.ArgMaxLastAxis (stream, trgt: NativeTensor, src: NativeTensor) =         
        let initial = minValueOf dataType
        argMaxLastAxis (stream, workDimForElemwise trgt, [|initial; box trgt; box src|])
        
    member this.FindLastAxis (value: obj) (stream, trgt: NativeTensor, src: NativeTensor) =         
        findLastAxis (stream, workDimForElemwise trgt, [|value; box trgt; box src|])        

    static member Get (dataType, nDims) = instances.Get (dataType, nDims)



type internal TensorGatherScatterKernels private (dataType: Type, nTrgtDims: int, nSrcDims: int) as this =
    inherit CudaModule()
    static let instances = InstanceCache TensorGatherScatterKernels 
    static let headers = ["GatherScatter.cuh"]

    let trgtTensor = ArgTypeTensor {DataType=dataType; NDims=nTrgtDims}
    let trgtIdxs = ArgTypeIdxTensors {NDims=nSrcDims; NIdxs=nTrgtDims}
    let srcTensor = ArgTypeTensor {DataType=dataType; NDims=nSrcDims}
    let srcIdxs = ArgTypeIdxTensors {NDims=nTrgtDims; NIdxs=nSrcDims}
    let ptrArg = ArgTypeScalar typeof<nativeint>
    let boolArg = ArgTypeScalar typeof<bool>

    let error = new CudaDeviceVariable<int32> (SizeT 1)
    do error.Memset (0u)
    let errorPtr = Cuda.getIntPtr error.DevicePointer

    let gather = this.GetKernel "Gather" [trgtTensor; srcIdxs; srcTensor; ptrArg; boolArg]
    let scatter = this.GetKernel "Scatter" [trgtTensor; trgtIdxs; srcTensor; ptrArg; boolArg]

    do this.Build (headers)

    member this.Gather (stream, trgt: NativeTensor, srcIdxs: NativeIdxTensors, src: NativeTensor) =         
        let trapOnError = not Cfg.Stacktrace
        gather (stream, workDimForElemwise trgt, [|box trgt; box srcIdxs; box src; 
                                                   box errorPtr; box trapOnError|])
        this.CheckError (stream)

    member this.Scatter (stream, trgt: NativeTensor, trgtIdxs: NativeIdxTensors, src: NativeTensor) =         
        let trapOnError = not Cfg.Stacktrace
        scatter (stream, workDimForElemwise src, [|box trgt; box trgtIdxs; box src; 
                                                   box errorPtr; box trapOnError|])
        this.CheckError (stream)

    member this.CheckError (stream) =
        if Cfg.Stacktrace then
            if stream <> CUstream.NullStream then
                Cuda.context.Synchronize()
            let hasError = ref 0
            error.CopyToHost (hasError)
            if !hasError <> 0 then
                raise (IndexOutOfRangeException "invalid index during gather or scatter")

    static member Get (dataType, nTrgtDims, nSrcDims) = 
        instances.Get (dataType, nTrgtDims, nSrcDims)



type internal TensorConvertKernels private (trgtDataType: Type, srcDataType: Type, nDims: int) as this =
    inherit CudaModule()
    static let instances = InstanceCache TensorConvertKernels 
    static let headers = ["Elemwise.cuh"]

    let trgtTensor = ArgTypeTensor {DataType=trgtDataType; NDims=nDims}
    let srcTensor = ArgTypeTensor {DataType=srcDataType; NDims=nDims}

    let convert = this.GetKernel "Convert" [trgtTensor; srcTensor]

    do this.Build (headers)

    member this.Convert (stream, trgt: NativeTensor, src: NativeTensor) =         
        convert (stream, workDimForElemwise trgt, [|box trgt; box src|])

    static member Get (trgtDataType, srcDataType, nDims) = 
        instances.Get (trgtDataType, srcDataType, nDims)



type internal BlasSupportKernels private () as this =
    inherit CudaModule()
    static let mutable instance = None 
    static let headers = ["BlasSupport.cuh"]

    let error = new CudaDeviceVariable<int32> (SizeT 1)
    do error.Memset (0u)
    let errorPtr = Cuda.getIntPtr error.DevicePointer

    let checkBlasInfo = 
        this.GetKernel "CheckBlasInfo" 
            [ArgTypeScalar typeof<nativeint>; ArgTypeScalar typeof<int>; 
             ArgTypeScalar typeof<nativeint>; ArgTypeScalar typeof<bool>]

    do this.Build (headers)

    member this.CheckBlasInfo (stream, info: CudaDeviceVariable<int>) = 
        let trapOnError = not Cfg.Stacktrace
        let infoPtr = Cuda.getIntPtr info.DevicePointer
        let batchSize = int info.Size
        let workDim = (int64 batchSize, 1L, 1L)
        checkBlasInfo (stream, workDim, [|box infoPtr; box batchSize; box errorPtr; box trapOnError|])

        if Cfg.Stacktrace then
            if stream <> CUstream.NullStream then Cuda.context.Synchronize()
            let hasError = ref 0
            error.CopyToHost (hasError)
            !hasError = 0 
        else
            true

    static member Get () = 
        match instance with
        | Some instance -> instance
        | None ->
            let inst = BlasSupportKernels()
            instance <- Some inst
            inst
      
