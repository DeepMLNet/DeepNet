open System
open System.IO
open System.Text.RegularExpressions

open YamlDotNet
open YamlDotNet.RepresentationModel

type Member = {
    Signature: string
    Name: string
    Summary: string
}

// Okay, what to do here?
// first, we definitely need links
// seconds the element-wise functions are a bit sucky, because they should not be called directly
// thirdly, we need operators, especially the comparison thingies
// big problem is how to document the operators...
// actually, it does not even make sense to link to the member functions, does it???
// Well it is actually okay, when in the sample we call them directly
// Problem with overrides is that documentation then becomes solely docfx accessible which might be undesired.
// so best keep documentation in source as much as possible.
// Put in source for now.
// Try writing exemplary documentation for Abs.


let sections = [
    "Elementwise functions", 
    "These mathematical functions are applied element-wise to each element of the tensor.", [
        "Abs"; "Acos"; "Asin"; "Atan"; "Ceiling"; "Cos"; "Cosh"; "Exp"; "Floor"; "Log"; "Log10"; 
        "Pow"; "Round"; "Sgn"; "Sin"; "Sinh"; "Sqrt"; "Tan"; "Tanh"; "Truncate"]
    
    "Shape functions", 
    "The following functions are for working with the shape and memory layout of a tensor.", [
        "atLeast1D"; "atLeast2D"; "atLeast3D"; "atLeastND"; "broadcastDim"; "broadcastTo"; 
        "broadcastToSame"; "broadcastToSameInDims"; "canReshapeView"; "CheckAxis"; "cutLeft"; "cutRight"; 
        "flatten"; "insertAxis"; "isBroadcasted"; "isColumnMajor"; "isRowMajor"; "layout"; "nDims"; 
        "nElems"; "offset"; "padLeft"; "padRight"; "padToSame"; "padToSame"; "padToSame"; "permuteAxes"; 
        "reshape"; "reshapeView"; "reverseAxis"; "relayout"; "shape"; "stride"; "swapDim"; "transpose"; 
        "tryReshapeView"]
    "Data type functions", "", [
        "bool"; "byte"; "convert"; "double"; "float"; "float32"; "int"; "int16"; "int32"; "int64"; 
        "nativeint"; "sbyte"; "single"; "uint16"; "uint32"; "uint64"; "dataType"]
    "Logical functions", "", [
        "all"; "allAxis"; "allElems"; "allTensor"; "any"; "anyAxis"; "anyTensor"; "allIdx"; 
        "allIdxOfDim"; "ifThenElse"]
    "Index functions", "", [
        "find"; "findAxis"; "gather"; "range"; "scatter"; "tryFind"; "trueIdx"]
    "Comparison functions", "", [
        "almostEqual"; "almostEqualWithTol"; "isClose"; "isCloseWithTol"; "isFinite"; 
        "maxElemwise"; "minElemwise"; "allFinite"]
    "Creation functions", "", [
        "arange"; "concat"; "copy"; "Copy"; "counting"; "empty"; "falses"; "identity"; 
        "diagMat"; "diagMatAxis"; "NewOfType"; "ofBlocks"; "ones"; "onesLike"; "init"; 
        "linspace"; "replicate"; "scalar"; "scalarLike"; "trues"; "zeros"; "zerosLike"]
    "Reduction functions", "", [
        "argMax"; "argMaxAxis"; "argMin"; "argMinAxis"; "countTrue"; "countTrueAxis";
        "countTrueTensor"; "max"; "maxAxis"; "maxTensor"; "min"; "minAxis"; "minTensor"; 
        "mean"; "meanAxis"; "product"; "productAxis"; "productTensor"; "std"; "stdAxis"; 
        "sum"; "sumAxis"; "sumTensor"; "var"; "varAxis"; "trace"; "traceAxis"]
    "Linear algebra functions", "", [
        "norm"; "normAxis"; "invert"; "pseudoInvert"; "SVD"; "SVDWithoutUV"; 
        "symmetricEigenDecomposition"]
    "Device functions", "", [
        "dev"; "transfer"; "TransferFrom"]
    "Tensor operations", "", [
        "diag"; "diagAxis"; "diff"; "diffAxis"; "dot"; "tensorProduct"]
    "Functional functions", "", [
        "foldAxis"; "map"; "map2"; "mapi"; "mapi2"]
    "Element access functions", "", [
        "get"; "set"; "value"]
]


[<EntryPoint>]
let main argv =
    let ys = YamlStream()
    use file = new StreamReader("../api/Tensor.Tensor-1.yml")
    ys.Load(file)
    let mapping = ys.Documents.[0].RootNode  :?> YamlMappingNode
    
    let entries = seq {
        for entry in mapping.Children.[YamlScalarNode "items"] :?> YamlSequenceNode do
            let entry = entry :?> YamlMappingNode
            let signature = (entry.Children.[YamlScalarNode "id"] :?> YamlScalarNode).Value
            if entry.Children.ContainsKey (YamlScalarNode "summary") then
                let summary = (entry.Children.[YamlScalarNode "summary"] :?> YamlScalarNode).Value
                let summary = summary.Replace("\n", " ").Trim()
                let name = 
                    match signature.IndexOf '(' with
                    | p when p >= 0 -> signature.[0 .. p-1]
                    | _ -> signature
                yield {Signature=signature; Name=name; Summary=summary}
    }

    //for entry in entries do
    //    printfn "Name: %s\nSignature: %s\nSummary: %s\n\n" entry.Name entry.Signature entry.Summary

    let findEntry (name: string) =
        entries |> Seq.find (fun entry -> entry.Name.ToLowerInvariant() = name.ToLowerInvariant())

    printfn "# Tensor"
    printfn "This page lists all tensor functions by category."
    printfn "For an alphabetical reference see [Tensor`1](Tensor)."
    printfn ""
    
    for title, descr, members in sections do
        printfn "## %s" title
        printfn "%s" descr
        printfn ""
        printfn "Function | Description"
        printfn "-------- | -----------"
        for name in members do
            let ent = findEntry name
            printfn "%s | %s" ent.Name ent.Summary
        printfn ""
        printfn ""



    0
