namespace SymTensor.Ops

type internal OpForwards() =
    interface IOpForwards with
        member this.DoBroadcast shp x = raise (System.NotImplementedException())
        member this.Reshape shp x = raise (System.NotImplementedException())
        member this.Negate x = {Negate.X=x} :> IOp2
        member this.Add x y = {Add.X=x; Y=y} :> IOp2

            