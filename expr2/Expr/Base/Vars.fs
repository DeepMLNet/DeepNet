namespace ExprNS

open ShapeSpec


module VarSpec =
    /// variable specification: has a name, type and shape specificaiton
    type VarSpecT<'T> = {Name: string;
                         Shape: ShapeSpecT;}

    /// create variable specifation by name and shape
    let inline ofNameAndShape name shape =
        {Name=name; Shape=shape;}

    /// name of variable
    let name (vs: VarSpecT<_>) = vs.Name

    /// shape of variable
    let shape (vs: VarSpecT<_>) = vs.Shape

