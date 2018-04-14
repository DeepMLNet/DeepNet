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
    "Operators",
    "These unary or binary operators can be applied to tensors.", [
        "( ~- )/op_UnaryNegation"; "( + )/op_Addition"; "( - )/op_Subtraction"; "( * )/op_Multiply"; 
        "( / )/op_Division"; "( % )/op_Modulus"; "( .* )/op_DotMultiply"]
    "Elementwise functions", 
    "These mathematical functions are applied element-wise to each element of the tensor.", [
        "Abs"; "Acos"; "Asin"; "Atan"; "Ceiling"; "Cos"; "Cosh"; "Exp"; "Floor"; "Log"; "Log10"; 
        "Pow"; "Round"; "Sgn"; "Sin"; "Sinh"; "Sqrt"; "Tan"; "Tanh"; "Truncate"]
    "Shape functions", 
    "The following functions are for working with the shape and memory layout of a tensor.", [
        "atLeast1D"; "atLeast2D"; "atLeast3D"; "atLeastND"; "broadcastDim"; "broadcastTo"; 
        "broadcastToSame"; "broadcastToSameInDims"; "CheckAxis"; "cutLeft"; "cutRight"; 
        "flatten"; "insertAxis"; "isBroadcasted"; "layout"; "nDims"; 
        "nElems"; "padLeft"; "padRight"; "padToSame"; "permuteAxes"; 
        "reshape"; "reshapeView"; "reverseAxis"; "relayout"; "shape"; "swapDim"; "transpose"; 
        "tryReshapeView"]
    "Data type functions", "", [
        "convert"; "dataType"]
    "Logical functions", "", [
        "( ~~~~ )/op_TwiddleTwiddleTwiddleTwiddle"; "( &&&& )/op_AmpAmpAmpAmp"; 
        "( \\|\\|\\|\\| )/op_BarBarBarBar"; "( ^^^^ )/op_HatHatHatHat";
        "all"; "allAxis"; "allElems"; "allTensor"; "any"; "anyAxis"; "anyTensor"; "allIdx"; 
        "allIdxOfDim"; "ifThenElse"]
    "Index functions", "", [
        "find"; "findAxis"; "gather"; "range"; "scatter"; "tryFind"; "trueIdx"]
    "Comparison functions", "", [
        "( ==== )/op_EqualsEqualsEqualsEquals"; "( <<<< )/op_LessLessLessLess"; 
        "( <<== )/op_LessLessEqualsEquals"; "( >>>> )/op_GreaterGreaterGreaterGreater"; 
        "( >>== )/op_GreaterGreaterEqualsEquals";
        "almostEqual"; "isClose"; "isFinite"; "maxElemwise"; "minElemwise"; "allFinite"]
    "Creation functions", "", [
        "arange"; "concat"; "copy"; "Copy"; "counting"; "empty"; "falses"; "identity"; 
        "diagMat"; "diagMatAxis"; "NewOfType"; "ofBlocks"; "ones"; "onesLike";  
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
//    "Functional functions", "", [
//        "foldAxis"; "map"; "map2"; "mapi"; "mapi2"]
    "Element access functions", "", [
        "get"; "set"; "value"]
]


[<EntryPoint>]
let main _ =
    use apiFile = new StreamReader "../api/Tensor.Tensor-1.yml"
    use outFile = new StreamWriter "../articles/Tensor.md"

    let out fmt = Printf.kprintf (fun s -> outFile.WriteLine s) fmt

    let ys = YamlStream()
    ys.Load(apiFile)
    let mapping = ys.Documents.[0].RootNode  :?> YamlMappingNode   
    let entries = seq {
        for entry in mapping.Children.[YamlScalarNode "items"] :?> YamlSequenceNode do
            let entry = entry :?> YamlMappingNode
            let signature = (entry.Children.[YamlScalarNode "id"] :?> YamlScalarNode).Value
            if entry.Children.ContainsKey (YamlScalarNode "summary") then
                let summary = (entry.Children.[YamlScalarNode "summary"] :?> YamlScalarNode).Value
                let summary = summary.Replace("\n", " ").Trim()
                let name = 
                    match signature.LastIndexOf '(' with
                    | p when p >= 0 -> signature.[0 .. p-1]
                    | _ -> signature
                yield {Signature=signature; Name=name; Summary=summary}
    }

    //for entry in entries do
    //    printfn "Name: %s\nSignature: %s\nSummary: %s\n\n" entry.Name entry.Signature entry.Summary

    let findEntry (name: string) =
        entries |> Seq.tryFind (fun entry -> entry.Name.ToLowerInvariant() = name.ToLowerInvariant())

    out "# Tensor"
    out "This page lists all tensor functions by category."
    out "For an alphabetical reference see [Tensor<'T>](xref:Tensor.Tensor`1)."
    out ""
    
    for title, descr, members in sections do
        out "## %s" title
        out "%s" descr
        out ""
        out "Function | Description"
        out "-------- | -----------"
        for name in members do
            let dispName, name = 
                match name.LastIndexOf "/" with
                | -1 -> name, name
                | p -> name.[..p-1], name.[p+1..]
            match findEntry name with
            | Some ent -> out "[%s](xref:Tensor.Tensor`1.%s*) | %s" dispName ent.Name ent.Summary
            | None -> out "%s | NOT FOUND" name
        out ""
        out ""



    0
