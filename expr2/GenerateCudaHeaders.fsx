open System.Text
open System.IO

let maxDims = 5

let sb = StringBuilder()
let sw = new StringWriter(sb)

let combineWith sep items =    
    let rec combine items = 
        match items with
        | [item] -> item
        | item::rest -> item + sep + combine rest
        | [] -> ""
    items |> Seq.toList |> combine

for dims = 0 to maxDims do
    let ad = {0 .. dims-1}

    fprintfn sw "template <%s>" 
        (ad |> Seq.map (fun i -> sprintf "size_t shape%d" i) |> combineWith ", ")
    fprintfn sw "class Shape%dD {" dims
    fprintfn sw "public:"
    fprintfn sw "  	_dev static size_t shape(const size_t dim) {"
    fprintfn sw "      switch (dim) {"
    for d in ad do
        fprintfn sw "        case %d: return shape%d;" d d
    fprintfn sw "        default: return 0;"
    fprintfn sw "      }"
    fprintfn sw "};"
    fprintfn sw ""

    fprintfn sw "template <%s>" 
        (ad |> Seq.map (fun i -> sprintf "size_t stride%d" i) |> combineWith ", ")
    fprintfn sw "class Stride%dD {" dims
    fprintfn sw "public:"
    fprintfn sw "  	_dev static size_t stride(const size_t dim) {"
    fprintfn sw "      switch (dim) {"
    for d in ad do
        fprintfn sw "        case %d: return stride%d;" d d
    fprintfn sw "        default: return 0;"
    fprintfn sw "      }"
    fprintfn sw "    }"
    fprintfn sw ""
    fprintfn sw "  	_dev static size_t offset(%s) {"
        (ad |> Seq.map (fun i -> sprintf "const size_t pos%d" i) |> combineWith ", ")
    fprintfn sw "      return %s;"
        (ad |> Seq.map (fun i -> sprintf "stride%d * pos%d" i i) |> combineWith " + ")
    fprintfn sw "    }"
    fprintfn sw "};"
    fprintfn sw ""

    fprintfn sw "template <typename TShape, typename TStride>"
    fprintfn sw "class NDArray%dD {" dims
    fprintfn sw "public:"
    fprintfn sw "  typedef TShape Shape;"
    fprintfn sw "  typedef TStride Stride;"
    fprintfn sw "  _dev static size_t shape(const size_t dim) { return Shape::shape(dim); }"
    fprintfn sw "  _dev static size_t stride(const size_t dim) { return Stride::stride(dim); }"
    fprintfn sw "  _dev static size_t size() {"
    fprintfn sw "    return %s;"
        (ad |> Seq.map (fun i -> sprintf "shape(%d)" i) |> combineWith " * ")
    fprintfn sw "  }"
    fprintfn sw "  _dev float *data() { return reinterpret_cast<float *>(this); }"
    fprintfn sw "  _dev const float *data() const { return reinterpret_cast<const float *>(this); }"
    fprintfn sw "  _dev float &element(%s) {"
        (ad |> Seq.map (fun i -> sprintf "size_t pos%d" i) |> combineWith ", ")
    fprintfn sw "    return data()[Stride::offset(%s)];"
        (ad |> Seq.map (fun i -> sprintf "pos%d" i) |> combineWith ", ")
    fprintfn sw "  }"
    fprintfn sw "  _dev const float &element(%s) const {"
        (ad |> Seq.map (fun i -> sprintf "size_t pos%d" i) |> combineWith ", ")
    fprintfn sw "    return data()[Stride::offset(%s)];"
        (ad |> Seq.map (fun i -> sprintf "pos%d" i) |> combineWith ", ")
    fprintfn sw "  }"
    fprintfn sw "};"
    fprintfn sw ""



let generated = sb.ToString()
printfn "%s" generated

