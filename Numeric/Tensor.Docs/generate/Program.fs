open System
open System.IO
open System.Text.RegularExpressions

open YamlDotNet
open YamlDotNet.RepresentationModel

type Member = {
    Prefix: string
    Signature: string
    Name: string
    Summary: string
}


let sections = [
    "Creation functions", 
    "Use these functions to create a new tensor.", [
        "arange"; "counting"; "empty"; "falses"; "filled"; "identity"; 
        "ones"; "onesLike";  "linspace"; "scalar"; "scalarLike"; "trues"; 
        "zeros"; "zerosLike"]

    "Slicing and element access functions", 
    "Use these functions to slice tensors or access individual elements of them.", [
        "Item"; "M"; "Value"]

    "Element-wise operations", 
    "These mathematical operations are applied element-wise to each element of the tensor(s).", [
        "( ~- )/op_UnaryNegation"; "( + )/op_Addition"; "( - )/op_Subtraction"; "( * )/op_Multiply"; 
        "( / )/op_Division"; "( % )/op_Modulus";         
        "Abs"; "Acos"; "Asin"; "Atan"; "Ceiling"; "Cos"; "Cosh"; "Exp"; "Floor"; "Log"; "Log10"; 
        "Pow"; "Round"; "Sgn"; "Sin"; "Sinh"; "Sqrt"; "Tan"; "Tanh"; "Truncate"]

    "Tensor operations", 
    "These functions perform various operations on one or more tensors.", [
        "concat"; "copy"; "diag"; "diagAxis"; "diagMat"; "diagMatAxis"; "diff"; "diffAxis"; 
        "ofBlocks"; "replicate"; "T"; ]        

    "Linear algebra functions", 
    "Use these functions to perform basic linear algebra operations on tensors.", [
        "( .* )/op_DotMultiply"; "norm"; "normAxis"; "invert"; "pseudoInvert"; 
        "SVD"; "symmetricEigenDecomposition"; "tensorProduct"]

    "Shape functions", 
    "Use these functions to work with the shape and memory layout of a tensor.", [
        "atLeastND"; "broadcastDim"; "broadcastTo"; 
        "broadcastToSame"; "broadcastToSameInDims"; "cutLeft"; "cutRight"; 
        "flatten"; "insertAxis"; "isBroadcasted"; "Layout"; "NDims"; 
        "NElems"; "padLeft"; "padRight"; "padToSame"; "permuteAxes"; 
        "reshape"; "reverseAxis"; "relayout"; "Shape"; "swapDim"; ]

    "Data type functions", 
    "Use these functions to query or change the data type of the elements of a tensor.", [
        "convert"; "DataType"]

    "Device and storage functions", 
    "Use these functions to query or change the storage device of a tensor.", [
        "Dev"; "Storage"; "transfer"]        

    "Comparison functions", 
    "Use these functions to perform comparisons of tensors. The results are mostly boolean tensors.", [
        "( ==== )/op_EqualsEqualsEqualsEquals"; "( <<<< )/op_LessLessLessLess"; 
        "( <<== )/op_LessLessEqualsEquals"; "( >>>> )/op_GreaterGreaterGreaterGreater"; 
        "( >>== )/op_GreaterGreaterEqualsEquals"; "( <<>> )/op_LessLessGreaterGreater";
        "almostEqual"; "isClose"; "isFinite"; "maxElemwise"; "minElemwise"; "allFinite"]

    "Logical functions", 
    "Use these functions to work with boolean tensors.", [
        "( ~~~~ )/op_TwiddleTwiddleTwiddleTwiddle"; "( &&&& )/op_AmpAmpAmpAmp"; 
        "( \\|\\|\\|\\| )/op_BarBarBarBar"; "( ^^^^ )/op_HatHatHatHat";
        "all"; "allAxis"; "allElems"; "allTensor"; "any"; "anyAxis"; "anyTensor"; 
        "ifThenElse"]

    "Index functions", 
    "These functions return tensor of indices or work with them.", [
        "allIdx"; "allIdxOfDim"; "argMax"; "argMaxAxis"; "argMin"; "argMinAxis";
        "find"; "findAxis"; "gather"; "scatter"; "trueIdx"]

    "Reduction functions", 
    "These functions perform operations on tensors that reduce their dimensionality.", [
        "countTrue"; "countTrueAxis"; "max"; "maxAxis"; "min"; "minAxis";
        "mean"; "meanAxis"; "product"; "productAxis"; "std"; "stdAxis"; 
        "sum"; "sumAxis"; "var"; "varAxis"; "trace"; "traceAxis"]

    "Functional operations (host only)", 
    "Use these functions to perform operations that are common in functional programming languages. \
     They require the tensor to be stored in host memory.", [
        "HostTensor.foldAxis"; "HostTensor.init"; "HostTensor.map"; "HostTensor.map2"; 
        "HostTensor.mapi"; "HostTensor.mapi2"]

    "Data exchange (host only)", 
    "Use these functions to convert tensors to and from other storage modalities. \
     They require the tensor to be stored in host memory.", [
        "HostTensor.ofArray"; "HostTensor.ofList"; "HostTensor.ofSeq";
        "HostTensor.read"; "HostTensor.readUntyped";
        "HostTensor.toArray"; "HostTensor.toList"; "HostTensor.toSeq";
        "HostTensor.usingArray"; 
        "HostTensor.write"]

    "Random number generation (host only)", 
    "Use these functions generate tensors filled with random numbers.", [
        "HostTensor.randomInt"; "HostTensor.randomNormal"; "HostTensor.randomUniform"]
]

let yamlEntries (prefix: string) (filename: string) = 
    use apiFile = new StreamReader (filename)
    let ys = YamlStream()
    ys.Load(apiFile)
    let mapping = ys.Documents.[0].RootNode :?> YamlMappingNode   
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
                yield {Prefix=prefix; Signature=signature; Name=name; Summary=summary}
    }
    Seq.cache entries


[<EntryPoint>]
let main _ =
    let tensor = yamlEntries "Tensor`1" "../api/Tensor.Tensor-1.yml"
    let hostTensor = yamlEntries "HostTensor" "../api/Tensor.HostTensor.yml"
    let cudaTensor = yamlEntries "CudaTensor" "../api/Tensor.CudaTensor.yml"
    let entries = Seq.concat [tensor; hostTensor; cudaTensor]

    //for entry in entries do
    //    printfn "Name: %s\nSignature: %s\nSummary: %s\n\n" entry.Name entry.Signature entry.Summary

    let findEntry (name: string) =
        let prefix, name =
            match name.LastIndexOf '.' with
            | -1 -> "Tensor`1", name
            | p -> name.[..p-1], name.[p+1..]
        entries |> Seq.tryFind (fun entry -> entry.Prefix.ToLowerInvariant() = prefix.ToLowerInvariant() &&
                                             entry.Name.ToLowerInvariant() = name.ToLowerInvariant())

    use outFile = new StreamWriter "../articles/Tensor.md"
    let out fmt = Printf.kprintf (fun s -> outFile.WriteLine s) fmt

    out "# Tensor"
    out "This page provides an overview of most commonly used tensor functions by category."
    out ""
    out "For a complete, alphabetical reference of all tensor functions see [Tensor<'T>](xref:Tensor.Tensor`1) \
         and the device-specific functions in [HostTensor](xref:Tensor.HostTensor) and [CudaTensor](xref:Tensor.CudaTensor)."
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
            | Some ent -> out "[%s](xref:Tensor.%s.%s*) | %s" dispName ent.Prefix ent.Name ent.Summary
            | None -> out "%s | NOT FOUND" name
        out ""
        out ""



    0
