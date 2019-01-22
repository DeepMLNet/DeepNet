namespace SymTensor.Ops

open SymTensor

type internal OpForwards() =
    interface IOpForwards with
        member __.Scalar value = {Scalar.Value=value} :> IOp2
        member __.SizeValue value = {SizeValue.Value=value} :> IOp2
        member __.Var var = {VarArg.Var=var} :> IOp2
        member __.Reshape shp x = {Reshape.Shape=shp; X=x} :> IOp2
        member __.DoBroadcast shp x = {DoBroadcast.Shape=shp; X=x} :> IOp2
        member __.PermuteAxes perm x = {PermuteAxes.Permutation=perm; X=x} :> IOp2
        member __.Subtensor range x = {Subtensor.Range=range; X=x} :> IOp2
        member __.IsSubtensor expr = isSubtensor expr
        member __.SetSubtensor range x y = {SetSubtensor.Range=range; X=x; Y=y} :> IOp2
        member __.Channel channel x = {Channel.Channel=channel; X=x} :> IOp2

        member __.UnaryPlus x = {UnaryPlus.X = x} :> IOp2
        member __.Negate x = {Negate.X = x} :> IOp2
        member __.Abs x = {Abs.X = x} :> IOp2
        member __.SignT x = {SignT.X = x} :> IOp2
        member __.Log x = {Log.X = x} :> IOp2
        member __.Log10 x = {Log10.X = x} :> IOp2
        member __.Exp x = {Exp.X = x} :> IOp2
        member __.Sin x = {Sin.X = x} :> IOp2
        member __.Cos x = {Cos.X = x} :> IOp2
        member __.Tan x = {Tan.X = x} :> IOp2
        member __.Asin x = {Asin.X = x} :> IOp2
        member __.Acos x = {Acos.X = x} :> IOp2
        member __.Atan x = {Atan.X = x} :> IOp2
        member __.Sinh x = {Sinh.X = x} :> IOp2
        member __.Cosh x = {Cosh.X = x} :> IOp2
        member __.Tanh x = {Tanh.X = x} :> IOp2
        member __.Sqrt x = {Sqrt.X = x} :> IOp2
        member __.Ceiling x = {Ceiling.X = x} :> IOp2
        member __.Floor x = {Floor.X = x} :> IOp2
        member __.Round x = {Round.X = x} :> IOp2
        member __.Truncate x = {Truncate.X = x} :> IOp2
        member __.Not x = {Not.X = x} :> IOp2
        member __.Store var x = {Store.Var=var; X=x} :> IOp2

        member __.Add x y = {Add.X = x; Y = y} :> IOp2
        member __.Subtract x y = {Subtract.X = x; Y = y} :> IOp2
        member __.Multiply x y = {Multiply.X = x; Y = y} :> IOp2
        member __.Divide x y = {Divide.X = x; Y = y} :> IOp2
        member __.Pow x y = {Pow.X = x; Y = y} :> IOp2
        member __.Modulo x y = {Modulo.X = x; Y = y} :> IOp2
        member __.And x y = {And.X = x; Y = y} :> IOp2
        member __.Or x y = {Or.X = x; Y = y} :> IOp2
        member __.Xor x y = {Xor.X = x; Y = y} :> IOp2
        member __.Equal x y = {Equal.X = x; Y = y} :> IOp2
        member __.NotEqual x y = {NotEqual.X = x; Y = y} :> IOp2
        member __.Less x y = {Less.X = x; Y = y} :> IOp2
        member __.LessOrEqual x y = {LessOrEqual.X = x; Y = y} :> IOp2
        member __.Greater x y = {Greater.X = x; Y = y} :> IOp2
        member __.GreaterOrEqual x y = {GreaterOrEqual.X = x; Y = y} :> IOp2
        member __.Dot x y = dot x y
