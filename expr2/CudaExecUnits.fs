module CudaExecUnits

open Util
open Op
open ExecUnitsGen

/// template instantiation specification
type TmplInstT = {FuncName: string; TmplArgs: string list; 
                  RetType: string; ArgTypes: string list;}

/// dimensionality of parallel work to perform
type WorkDimT = int * int * int

/// a CUDA operation 
type CudaOpT =
    // memory operations
    | MemcpyDtoD of NDArrayViewT * NDArrayViewT
    | MemcpyHtoD of NDArrayViewT * NDArrayViewT
    | MemcpyDtoH of NDArrayViewT * NDArrayViewT
    | Memset of float * NDArrayViewT
    // kernel execution
    | LaunchKernel of TmplInstT * WorkDimT * (obj list)

/// converts sequence of ints to sequence of strings
let toStrSeq items =
    Seq.map (sprintf "%d") items

/// C++ NDArray type string for given NDArrayView
let cudaNDArrayCType view =
    let dims = NDArrayView.nDim view
    let shapeStr = if dims = 0 then "" else "<" + (view.Shape |> toStrSeq |> String.combineWith ",") + ">"
    let strideStr = "<" + ((view.Offset :: view.Stride) |> toStrSeq |> String.combineWith ",") + ">"
    sprintf "NDArray%dD<Shape%dD%s, Stride%dD%s > " dims dims shapeStr dims strideStr


/// computes the definitive target view of an op given its source views
let trgtViewGivenSrc memAllocator trgtShape reqView op srcViews srcShared  =
    // target that shares no elements with any srcView
    let outplaceTrgt =
        match reqView with
        | Some rv when not (List.exists (NDArrayView.overlapping rv) srcViews) -> rv, false
        | _ -> NDArrayView.newContinguous memAllocator trgtShape, false        

    // target that reuses a srcView, if it may be overwritten
    let inplaceOverwriteTrgt =
        match List.tryFindIndex not srcShared with
        | Some i -> srcViews.[i], false
        | None -> outplaceTrgt    

    match op with
    // variable access
    | LeafOp (Var vs) ->
        // TODO: use variable memory
        NDArrayView.newContinguous memAllocator trgtShape, false        
    // tensor creation
    | LeafOp _ -> outplaceTrgt        

    // unary elementwise
    | UnaryOp Negate -> inplaceOverwriteTrgt
    | UnaryOp Log -> inplaceOverwriteTrgt
    | UnaryOp Exp -> inplaceOverwriteTrgt
    // reductions
    | UnaryOp Sum -> outplaceTrgt
    | UnaryOp (SumAxis _) -> outplaceTrgt
    // shape operations
    | UnaryOp (Reshape _) ->        
        // TODO: optimize: check if copy is really necessary
        if NDArrayView.isContiguous srcViews.[0] then
            {srcViews.[0] with Shape=trgtShape; Stride=NDArrayView.contiguousStride trgtShape}, srcShared.[0]
        else outplaceTrgt  // will copy
    | UnaryOp (Broadcast _) ->
        let aView, aShared = srcViews.[0], srcShared.[0]
        {aView with Shape=trgtShape; 
                    Stride=List.map3 
                        (fun aStr aShp tShp -> if aShp = tShp then aStr else 0) 
                        aView.Stride aView.Shape trgtShape}, aShared
    | UnaryOp (SwapDim (ax1, ax2)) ->
        let aView, aShared = srcViews.[0], srcShared.[0]
        let str = aView.Stride
        {aView with Shape=trgtShape; 
                    Stride=str |> List.set ax1 str.[ax2] |> List.set ax2 str.[ax1]}, aShared
    // misc
    | UnaryOp (Annotated _) -> srcViews.[0], srcShared.[0]

    // binary elementwise
    | BinaryOp Add -> inplaceOverwriteTrgt
    | BinaryOp Substract -> inplaceOverwriteTrgt
    | BinaryOp Multiply -> inplaceOverwriteTrgt
    | BinaryOp Divide -> inplaceOverwriteTrgt
    | BinaryOp Power -> inplaceOverwriteTrgt
    // matrix/tensor operations
    | BinaryOp Dot -> outplaceTrgt
    | BinaryOp TensorProduct -> outplaceTrgt
      

/// computes desired source views given desired target view
let srcViewReqsGivenTrgt trgtShape reqView op srcShapes =
    let nSrcs = List.length srcShapes

    // requests all sources to use separate storage
    let outplaceTrgt =
        List.replicate nSrcs None

    // requests one source to be evaluated into our target view
    let inplaceOverwriteTrgt =
        match nSrcs with
        | 0 -> []
        | 1 -> [reqView]
        | _ -> reqView :: List.replicate (nSrcs-1) None

    match op with
    | LeafOp _ -> []

    // unary elementwise
    | UnaryOp Negate -> inplaceOverwriteTrgt
    | UnaryOp Log -> inplaceOverwriteTrgt
    | UnaryOp Exp -> inplaceOverwriteTrgt
    // reductions
    | UnaryOp Sum -> outplaceTrgt
    | UnaryOp (SumAxis _) -> outplaceTrgt
    // shape operations
    | UnaryOp (Reshape _) ->        
        match reqView with
        | Some rv when NDArrayView.isContiguous rv ->
            [Some {rv with Shape=srcShapes.[0]; Stride=NDArrayView.contiguousStride srcShapes.[0]}]
        | _ -> outplaceTrgt
    | UnaryOp (Broadcast _) -> outplaceTrgt
    | UnaryOp (SwapDim (ax1, ax2)) ->
        match reqView with
        | Some rv ->
            let str = rv.Stride
            [Some {rv with Shape=srcShapes.[0]; 
                           Stride=str |> List.set ax1 str.[ax2] |> List.set ax2 str.[ax1]}]
        | _ -> outplaceTrgt
    // misc
    | UnaryOp (Annotated _) -> inplaceOverwriteTrgt

    // binary elementwise
    | BinaryOp Add -> inplaceOverwriteTrgt
    | BinaryOp Substract -> inplaceOverwriteTrgt
    | BinaryOp Multiply -> inplaceOverwriteTrgt
    | BinaryOp Divide -> inplaceOverwriteTrgt
    | BinaryOp Power -> inplaceOverwriteTrgt
    // matrix/tensor operations
    | BinaryOp Dot -> outplaceTrgt
    | BinaryOp TensorProduct -> outplaceTrgt     


/// execution items for an elementwise operation
let execItemsForElemwise trgtView cOp cOpIndexed srcViews =
    let nSrc = List.length srcViews
    let argTypes = cudaNDArrayCType trgtView :: (List.map cudaNDArrayCType srcViews)
    let argTypesPointers = argTypes |> List.map (fun at -> at + " *")
    let indexedStr = if cOpIndexed then "Indexed" else ""
    let kernel = 
        {FuncName=sprintf "elementwise%dAry%dD%s" nSrc (NDArrayView.nDim trgtView) indexedStr;
         TmplArgs=cOp :: argTypes;
         RetType="void";
         ArgTypes=argTypesPointers}

    let workDim = 
        match NDArrayView.nDim trgtView with
        | 0 -> (1, 1, 1)
        | 1 -> (trgtView.Shape.[0], 1, 1)
        | 2 -> (trgtView.Shape.[0], trgtView.Shape.[1], 1)
        | 3 -> (trgtView.Shape.[0], trgtView.Shape.[1], trgtView.Shape.[2])
        | d ->
            let rest = {2 .. d-1} |> Seq.map (fun i -> trgtView.Shape.[i]) |> Seq.fold (*) 1 
            (trgtView.Shape.[0], trgtView.Shape.[1], rest)

    [LaunchKernel(kernel, workDim, (trgtView.Memory :> obj) :: (List.map (fun v -> v.Memory :> obj) srcViews))]


/// returns the execution units for the specified op
let execItemsForOp trgtView op srcViews =
    match op with 
    // tensor creation
    | LeafOp (DiagonalOne _) -> execItemsForElemwise trgtView "DiagonalOneIEOp_t" true []
    | LeafOp (Zeros _) -> execItemsForElemwise trgtView "ZerosEOp_t" true []
    | LeafOp (ScalarConst f) -> execItemsForElemwise trgtView (sprintf "ConstEOp_t<%f>" f) true []
    | LeafOp (TensorConst(f, _)) -> execItemsForElemwise trgtView (sprintf "ConstEOp_t<%f>" f) true []
    // variable access
    | LeafOp (Var vs) -> []
        
    // unary elementwise
    | UnaryOp Negate -> execItemsForElemwise trgtView "NegateEOp_t" false srcViews
    | UnaryOp Log -> execItemsForElemwise trgtView "LogEOp_t" false srcViews
    | UnaryOp Exp -> execItemsForElemwise trgtView "ExpEOp_t" false srcViews
    // reductions
    | UnaryOp Sum -> execItemsForElemwise trgtView "Sum" false srcViews // TODO
    | UnaryOp (SumAxis _) -> execItemsForElemwise trgtView "SumAxis" false srcViews // TODO
    // shape operations
    | UnaryOp (Reshape _) ->
        if trgtView <> srcViews.[0] then execItemsForElemwise trgtView "IdEOp_t" false srcViews
        else []
    | UnaryOp (Broadcast _) -> []
    | UnaryOp (SwapDim _) -> []
    // misc
    | UnaryOp (Annotated _) -> []

    // binary elementwise
    | BinaryOp Add -> execItemsForElemwise trgtView "AddEOp_t" false srcViews
    | BinaryOp Substract -> execItemsForElemwise trgtView "SubstractEOp_t" false srcViews
    | BinaryOp Multiply -> execItemsForElemwise trgtView "MultiplyEOp_t" false srcViews
    | BinaryOp Divide -> execItemsForElemwise trgtView "DivideEOp_t" false srcViews
    | BinaryOp Power -> execItemsForElemwise trgtView "PowerEOp_t" false srcViews
    // matrix/tensor operations
    | BinaryOp Dot -> execItemsForElemwise trgtView "Dot" false srcViews // TODO
    | BinaryOp TensorProduct -> execItemsForElemwise trgtView "TensorProduct" false srcViews // TODO


/// generates CUDA execution units that will evaluate the given unified expression
let exprToCudaExecUnits =
    exprToExecUnits execItemsForOp trgtViewGivenSrc srcViewReqsGivenTrgt 


