namespace SymTensor

[<AutoOpen>]
module ModelContextTypes =

    type MC (context: string) =

        let toSizeSpec (name: string) =
            let fullName = context + "." + name
            if name.StartsWith "_" then SizeSpec.flexSymbol fullName
            else SizeSpec.symbol fullName

        let toShapeSpec (shapeObj: obj list) =
            shapeObj |> List.map
                (function
                | :? string as symName -> toSizeSpec symName
                | :? int as f when f = -1 -> SizeSpec.broadcastable
                | :? int as f when f >= 0 -> SizeSpec.fix f
                | r -> failwithf "size must be either a size symbol name (string), \
                                  a fixed size (positive integer) or -1 for broadcast, but got %A" r)

        new() = MC "root"

        member this.Var (name: string) (shape: obj list) =
             Expr.var (context + "." + name) (toShapeSpec shape)

        member this.Size (name: string) =
            toSizeSpec name

        member this.Module (name: string) =
            MC (context + "." + name)
            


