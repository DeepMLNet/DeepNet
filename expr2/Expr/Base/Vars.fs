namespace ExprNS

open ShapeSpec


module VarSpec =

    type IVarSpec =
        inherit System.IComparable
        abstract member Name : string 
        abstract member Shape: ShapeSpecT

    /// variable specification: has a name, type and shape specificaiton
    type VarSpecT<'T> = 
        {Name: string; Shape: ShapeSpecT;}
        
        interface IVarSpec with
            member this.Name = this.Name
            member this.Shape = this.Shape

    /// create variable specifation by name and shape
    let inline ofNameAndShape name shape =
        {Name=name; Shape=shape;}

    /// name of variable
    let name (vs: IVarSpec) = vs.Name

    /// shape of variable
    let shape (vs: IVarSpec) = vs.Shape

