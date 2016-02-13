﻿open System.Text
open System.IO

let maxDims = 5
let maxArity = 2

let combineWith sep items =    
    let rec combine items = 
        match items with
        | [item] -> item
        | item::rest -> item + sep + combine rest
        | [] -> ""
    items |> Seq.toList |> combine

let combineWithButIfEmpty empty sep items =
    if Seq.isEmpty items then empty
    else combineWith sep items

let sw = new StreamWriter("NDSupport.cuh")
let prn = sprintf
let cw = combineWith
let cwe = combineWithButIfEmpty
let (|>>) seq mapFun = Seq.map mapFun seq

let wrt frmt = fprintfn sw frmt
 
wrt "// this file is automatically generated by GenerateNDSupport.fsx"
wrt "#pragma once"
wrt """#include "Utils.cuh" """
wrt ""  

for dims = 0 to maxDims do
    let ad = {0 .. dims-1}
    
    wrt "// ======================== dimensionality: %d ==================================" dims
    wrt ""

    wrt "struct Pos%dD {" dims
    wrt "   size_t pos[%d];" (max dims 1)
    wrt"    template<typename TNDArray>"
    wrt "   _dev static Pos%dD fromLinearIdx(size_t idx) {" dims
    wrt "     Pos%dD p;" dims
    if dims >= 1 then
        wrt "     const size_t incr0 = 1;"
    for d = 1 to dims - 1 do
        wrt "     const size_t incr%d = incr%d * TNDArray::shape(%d);" d (d-1) (d-1)
    for d = dims - 1 downto 0 do
        wrt "     p.pos[%d] = idx / incr%d;" d d
        wrt "     idx -= p.pos[%d] * incr%d;" d d
    if dims = 0 then
        // to silence compiler warning
        wrt "     p.pos[0] = 0;"
    wrt "     return p;"
    wrt "   }"
    wrt"    template<typename TNDArray>"
    wrt "   _dev static Pos%dD fromLinearIdxWithLastDimSetToZero(size_t idx) {" dims
    wrt "     Pos%dD p = fromLinearIdx<TNDArray>(idx);" dims 
    if dims >= 1 then
        wrt "     p[%d] = 0;" (dims - 1)
    wrt "     return p;"
    wrt "    }"
    wrt "    template<typename TNDArray>"
    wrt "   _dev size_t toLinearIdx() const {"
    if dims >= 1 then
        wrt "     const size_t incr0 = 1;"
    for d = 1 to dims - 1 do
        wrt "     const size_t incr%d = incr%d * TNDArray::shape(%d);" d (d-1) (d-1)
    wrt "     return %s;" (ad |>> (fun i -> prn "incr%d * pos[%d]" i i) |> cwe "0" " + ")
    wrt "   }"
    wrt "  	_dev size_t &operator[] (const size_t dim) { return pos[dim]; }"
    wrt "  	_dev const size_t &operator[] (const size_t dim) const { return pos[dim]; }"
    wrt "};"
    wrt ""
    
    if dims > 0 then
        wrt "template <%s>" (ad |>> prn "size_t shape%d" |> cw ", ")
    wrt "struct ShapeStatic%dD {" dims
    wrt "  	_dev static size_t shape(const size_t dim) {"
    if dims > 0 then
        wrt "      switch (dim) {"
        for d in ad do
            wrt "        case %d: return shape%d;" d d
        wrt "        default: return 0;"
        wrt "      }"
    else
        wrt "      return 0;"
    wrt "   }"
    wrt "};"
    wrt ""

    if dims = 0 then
        wrt "template <size_t offset_>"
    else
        wrt "template <size_t offset_, %s>" (ad |>> prn "size_t stride%d" |> cw ", ")

    wrt "struct StrideStatic%dD {" dims
    //wrt "public:"
    wrt "  	_dev static size_t stride(const size_t dim) {"
    wrt "      switch (dim) {"
    for d in ad do
        wrt "        case %d: return stride%d;" d d
    wrt "        default: return 0;"
    wrt "      }"
    wrt "    }"
    wrt "   _dev static size_t offset() {"
    wrt "      return offset_;"
    wrt "    }"
    wrt "  	_dev static size_t index(%s) {"
        (ad |>> prn "const size_t pos%d" |> cw ", ")
    wrt "      return offset_ + %s;"
        (ad |>> (fun i -> prn "stride%d * pos%d" i i) |> cwe "0" " + ")
    wrt "    }"
    wrt "  	_dev static size_t index(const size_t *pos) {"
    wrt "      return offset_ + %s;"
        (ad |>> (fun i -> prn "stride%d * pos[%d]" i i) |> cwe "0" " + ")
    wrt "    }"
    wrt "  	_dev static size_t index(const Pos%dD &pos) {" dims
    wrt "      return offset_ + %s;"
        (ad |>> (fun i -> prn "stride%d * pos[%d]" i i) |> cwe "0" " + ")
    wrt "    }"
    wrt "};"
    wrt ""

    wrt "template <typename TShape, typename TStride>"
    wrt "struct NDArrayStatic%dD {" dims
    wrt "  typedef TShape Shape;"
    wrt "  typedef TStride Stride;"
    wrt "  typedef Pos%dD Pos;" dims
    wrt "  float *mData;"
    wrt ""
    wrt "  _dev static size_t shape(const size_t dim) { return Shape::shape(dim); }"
    wrt "  _dev static size_t stride(const size_t dim) { return Stride::stride(dim); }"
    wrt "  _dev static size_t nDim() { return %d; }" dims
    wrt "  _dev static size_t nElems() { return Shape::nElems(); }"
    wrt "  _dev static size_t offset() { return Stride::offset(); }"
    wrt "  _dev static size_t size() {"
    wrt "    return %s;"
        (ad |>> prn "shape(%d)" |> cwe "1" " * ")
    wrt "  }"
    wrt "  _dev static Pos%dD linearIdxToPos(size_t idx) { return Pos%dD::fromLinearIdx<Shape>(idx); }" dims dims
    wrt "  _dev static Pos%dD linearIdxToPosWithLastDimSetToZero(size_t idx) { return Pos%dD::fromLinearIdxWithLastDimSetToZero<Shape>(idx); }" dims dims
    wrt "  _dev static size_t index(const size_t *pos) { return Stride::index(pos); }"
    wrt "  _dev static size_t index(const Pos%dD &pos) { return Stride::index(pos); }" dims
    wrt "  _dev float *data() { return mData; }"
    wrt "  _dev const float *data() const { return mData; }"
    wrt "  _dev float &element(%s) {"
        (ad |>> prn "size_t pos%d" |> cw ", ")
    wrt "    return data()[Stride::index(%s)];"
        (ad |>> prn "pos%d" |> cw ", ")
    wrt "  }"
    wrt "  _dev const float &element(%s) const {"
        (ad |>> prn "size_t pos%d" |> cw ", ")
    wrt "    return data()[Stride::index(%s)];"
        (ad |>> prn "pos%d" |> cw ", ")
    wrt "  }"
    wrt "  _dev float &element(const size_t *pos) {"
    wrt "    return data()[Stride::index(pos)];"
    wrt "  }"
    wrt "  _dev const float &element(const size_t *pos) const {"
    wrt "    return data()[Stride::index(pos)];"
    wrt "  }"
    wrt "  _dev float &element(const Pos%dD &pos) {" dims
    wrt "    return data()[Stride::index(pos)];"
    wrt "  }"
    wrt "  _dev const float &element(const Pos%dD &pos) const {" dims
    wrt "    return data()[Stride::index(pos)];"
    wrt "  }"
    wrt "};"
    wrt ""

    let elementwiseLoop withPosArray fBody =
        wrt "" 
        if dims > 3 then
            let restElements = 
                {2 .. dims - 1} |> Seq.map (sprintf "TTarget::shape(%d)") |> combineWith " * "
            wrt "    const size_t itersRest = (%s) / (gridDim.z * blockDim.z) + 1;" restElements
        if dims = 3 then
            wrt "    const size_t iters2 = TTarget::shape(2) / (gridDim.z * blockDim.z) + 1;"
        if dims >= 2 then
            wrt "    const size_t iters1 = TTarget::shape(1) / (gridDim.y * blockDim.y) + 1;"
        if dims >= 1 then
            wrt "    const size_t iters0 = TTarget::shape(0) / (gridDim.x * blockDim.x) + 1;"

        if dims > 3 then
            wrt "    for (size_t iterRest = 0; iterRest < itersRest; iterRest++) {"
        if dims = 3 then
            wrt "    for (size_t iter2 = 0; iter2 < iters2; iter2++) {"
        if dims >= 2 then
            wrt "    for (size_t iter1 = 0; iter1 < iters1; iter1++) {"
        if dims >= 1 then
            wrt "    for (size_t iter0 = 0; iter0 < iters0; iter0++) {"

        if dims > 3 then
            wrt "    size_t posRest = threadIdx.z + blockIdx.z * blockDim.z + iterRest * (gridDim.z * blockDim.z);"
            wrt "    const size_t incr2 = 1;"
            for d = 3 to dims - 1 do
                wrt "    const size_t incr%d = incr%d * TTarget::shape(%d);" d (d-1) (d-1)
            for d = dims - 1 downto 2 do
                wrt "    const size_t pos%d = posRest / incr%d;" d d
                wrt "    posRest -= pos%d * incr%d;" d d
        if dims = 3 then
            wrt "    const size_t pos2 = threadIdx.z + blockIdx.z * blockDim.z + iter2 * (gridDim.z * blockDim.z);"
        if dims >= 2 then
            wrt "    const size_t pos1 = threadIdx.y + blockIdx.y * blockDim.y + iter1 * (gridDim.y * blockDim.y);"
        if dims >= 1 then
            wrt "    const size_t pos0 = threadIdx.x + blockIdx.x * blockDim.x + iter0 * (gridDim.x * blockDim.x);"
    
            wrt "    if (%s) {"
                (ad |>> (fun i -> prn "(pos%d < trgt.shape(%d))" i i) |> cw " && ")

        if withPosArray then
            let poses = ad |> Seq.map (sprintf "pos%d")
            if dims >= 1 then
                wrt "    const size_t pos[] {%s};" (poses |> cw ", ")
            else
                wrt "    const size_t *pos = nullptr;"

        wrt ""
        fBody dims
        wrt ""

        if dims >= 1 then
            wrt "    }"

        if dims >= 1 then
            wrt "    }"
        if dims >= 2 then
            wrt "    }"
        if dims >= 3 then
            wrt "    }"   


    let elementwiseWrapper ary withIndexes =
        let srcTmpl = 
            {0 .. ary - 1} |> Seq.map (sprintf "typename TSrc%d") |> Seq.toList
        let allTmpl = "typename TTarget" :: srcTmpl
        wrt "template <typename TElemwiseOp, %s>" (allTmpl |> cw ", ")

        let srcArgDecls =
            {0 .. ary - 1} |> Seq.map (fun i -> sprintf "const TSrc%d &src%d" i i) |> Seq.toList
        let allArgDecls = "const TElemwiseOp &op" :: "TTarget &trgt" :: srcArgDecls
        let indexedName = if withIndexes then "Indexed" else ""
        wrt "_dev void elemwise%dAry%dD%s(%s) {" ary dims indexedName (allArgDecls |> cw ", ")

        elementwiseLoop withIndexes (fun dims ->      
            let poses = ad |>> prn "pos%d" |> cw ", "
            let srcArgs = {0 .. ary - 1} |> Seq.map (fun a -> sprintf "src%d.element(%s)" a poses) |> Seq.toList
            let allArgs = if withIndexes then "pos" :: sprintf "%d" dims :: srcArgs else srcArgs
            wrt "  trgt.element(%s) = op(%s);" poses (allArgs |> cw ", "))        
        wrt "}"
        wrt ""

    let elementwiseHeterogenousLoop fBody =
        wrt "" 
        wrt "    const size_t iters = TTarget::size() / (gridDim.x * blockDim.x) + 1;" 
        wrt "    for (size_t iter = 0; iter < iters; iter++) {"
        wrt "    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x + iter * (gridDim.x * blockDim.x);"   
        wrt "    if (idx < TTarget::size()) {"

        wrt ""
        fBody dims
        wrt ""

        wrt "    }"
        wrt "    }"

    let elementwiseHeterogenousWrapper ary =
        let srcTmpl = 
            {0 .. ary - 1} |> Seq.map (sprintf "typename TSrc%d") |> Seq.toList
        let allTmpl = "typename TTarget" :: srcTmpl
        wrt "template <typename TElemwiseOp, %s>" (allTmpl |> cw ", ")

        let srcArgDecls =
            {0 .. ary - 1} |> Seq.map (fun i -> sprintf "const TSrc%d &src%d" i i) |> Seq.toList
        let allArgDecls = "const TElemwiseOp &op" :: "TTarget &trgt" :: srcArgDecls
        wrt "_dev void elemwise%dAry%dDHeterogenous(%s) {" ary dims (allArgDecls |> cw ", ")        
        elementwiseHeterogenousLoop (fun dims ->      
            let srcArgs = 
                {0 .. ary - 1} 
                |> Seq.map (fun a -> sprintf "src%d.element(src%d.linearIdxToPos(idx))" a a) 
                |> Seq.toList
            let allArgs = srcArgs
            wrt "  trgt.element(trgt.linearIdxToPos(idx)) = op(%s);" (allArgs |> cw ", "))        
        wrt "}"
        wrt ""


    for ary = 0 to maxArity do
        for withIndexes in [true; false] do
            elementwiseWrapper ary withIndexes
        elementwiseHeterogenousWrapper ary

    ()


sw.Dispose()


