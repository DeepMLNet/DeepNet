namespace SymTensor.Deriv

open SymTensor
open SymTensor.Ops


[<OpExtender>]
type UnaryPlusDeriv(op: UnaryPlus) =
    interface IDerivableOp with
        member this.Deriv dOp = dOp |> Args.unary


[<OpExtender>]
type NegateDeriv(op: Negate) =
    interface IDerivableOp with
        member this.Deriv dOp = -dOp |> Args.unary


[<OpExtender>]
type AbsDeriv(op: Abs) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (Expr2.signt op.X) |> Args.unary


[<OpExtender>]
type SignTDeriv(op: SignT) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            Deriv.zeros dOp op.X |> Args.unary

            
[<OpExtender>]
type LogDeriv(op: Log) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let one = Consts.one op.X
            dOp * Expr2.padLeft (op.X ** (-one)) |> Args.unary

[<OpExtender>]
type Log10Deriv(op: Log10) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let one = Consts.one op.X
            let ten = Consts.ten op.X
            dOp * Expr2.padLeft (op.X ** (-one) / log ten) |> Args.unary


[<OpExtender>]
type ExpDeriv(op: Exp) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (exp op.X) |> Args.unary


[<OpExtender>]
type SinDeriv(op: Sin) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (cos op.X) |> Args.unary


[<OpExtender>]
type CosDeriv(op: Cos) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (-sin op.X) |> Args.unary

        
[<OpExtender>]
type TanDeriv(op: Tan) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let one = Consts.one op.X
            let two = Consts.two op.X
            dOp * Expr2.padLeft (one + (tan op.X) ** two) |> Args.unary


[<OpExtender>]
type AsinDeriv(op: Asin) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let one = Consts.one op.X
            let two = Consts.two op.X
            dOp * Expr2.padLeft (one / Expr2.sqrtt (one - op.X ** two)) |> Args.unary


[<OpExtender>]
type AcosDeriv(op: Acos) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let one = Consts.one this.X
            let two = Consts.two this.X
            dOp * Expr2.padLeft (-one / Expr2.sqrtt (one - op.X ** two)) |> Args.unary


[<OpExtender>]
type AtanDeriv(op: Atan) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let one = Consts.one op.X
            let two = Consts.two op.X
            dOp * Expr2.padLeft (one / (one + op.X ** two)) |> Args.unary 






[<OpExtender>]
type SinhDeriv(op: Sinh) =
    interface IDerivableOp with      
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (cosh op.X) |> Args.unary


[<OpExtender>]
type CoshDeriv(op: Cosh) =
    interface IDerivableOp with      
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (sinh op.X) |> Args.unary


[<OpExtender>]
type TanhDeriv(op: Tanh) =
    interface IDerivableOp with      
        member this.Deriv dOp = 
            let one = Consts.one op.X
            let two = Consts.two op.X
            dOp * Expr2.padLeft (one - (tanh op.X) ** two) |> Args.unary
        

[<OpExtender>]
type SqrtDeriv(op: Sqrt) =
    interface IDerivableOp with      
        member this.Deriv dOp = 
            let one = Consts.one op.X
            let two = Consts.two op.X
            dOp * Expr2.padLeft (one / (two * Expr2.sqrtt op.X)) |> Args.unary


[<OpExtender>]
type CeilingDeriv(op: Ceiling) =
    interface IDerivableOp with      
        member this.Deriv dOp = Consts.zeros dOp op.X |> Args.unary


[<OpExtender>]
type FloorDeriv(op: Floor) =
    interface IDerivableOp with      
        member this.Deriv dOp = Consts.zeros dOp op.X |> Args.unary


[<OpExtender>]
type RoundDeriv(op: Round) =
    interface IDerivableOp with      
        member this.Deriv dOp = Consts.zeros dOp op.X |> Args.unary


[<OpExtender>]
type TruncateDeriv(op: Truncate) =
    interface IDerivableOp with      
        member this.Deriv dOp = Consts.zeros dOp op.X |> Args.unary


[<OpExtender>]
type InvertDeriv(op: Invert) =
    interface IDerivableOp with      
        member this.Deriv dOp = 
            let self = this |> Expr2
            -(Expr2.padLeft self.T) .* dOp .* (Expr2.padLeft self.T) |> Args.unary 


[<OpExtender>]
type NotDeriv(op: Not) =
    interface IDerivableOp with      
        member this.Deriv dOp = Consts.zeros dOp op.X |> Args.unary


[<OpExtender>]
type ReshapeDeriv(op: Reshape) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let funElems = dOp.Shape.[0]
            dOp |> Expr2.reshape (funElems :: op.X.Shape) |> Args.unary


[<OpExtender>]
type DoBroadcastDeriv(op: DoBroadcast) =
    interface IDerivableOp with      
        member this.Deriv dOp = 
            let mutable dOpUnBc = dOp
            for ax, (bSize, xSize) in List.indexed (List.zip this.Shape op.X.Shape) do
                match bSize, xSize with
                | SizeSpec.Broadcast, SizeSpec.Broadcast -> ()
                | _, SizeSpec.Broadcast ->
                    dOpUnBc <- dOpUnBc |> sumKeepingAxis (ax + 1)
                | _ -> ()
            dOpUnBc |> Args.unary                 


[<OpExtender>]
type PermuteAxesDeriv(op: PermuteAxes) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let backPerm = Permutation.invert this.Permutation
            let dOpPerm = 
                0 :: List.map (fun p -> p + 1) backPerm
            dOp |> Expr2.permuteAxes dOpPerm |> Args.unary                 


[<OpExtender>]
type SubtensorDeriv(op: Subtensor) =
    interface IDerivableOp with      
        member this.Deriv dOp = 
            let funElems = dOp.Shape.[0]
            let agExpanded = Expr2.zerosOfType dOp.DataType (funElems :: op.X.Shape)
            Expr2.setSubtensor agExpanded.[SimpleRangeSpec.All :: this.Range] dOp
            |> Args.unary


[<OpExtender>]
type SetSubtensorDeriv(op: SetSubtensor) =
    interface IDerivableOp with      
        member this.Deriv dOp = 
            let dYExp = dOp.[SimpleRangeSpec.All :: this.Range]
            let zeros = Expr2.zerosOfType dYExp.DataType dYExp.Shape
            let dXExp = Expr2.setSubtensor dOp.[SimpleRangeSpec.All :: this.Range] zeros
            Args.binary dXExp dYExp


[<OpExtender>]
type ReverseAxisDeriv(op: ReverseAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            dOp |> reverseAxis (this.Axis + 1) |> Args.unary


[<OpExtender>]
type DiagDeriv(op: Diag) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type DiagMatDeriv(op: DiagMat) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type SumAxisDeriv(op: SumAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type ProductAxisDeriv(op: ProductAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type MaxAxisDeriv(op: MaxAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type MinAxisDeriv(op: MinAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type ArgMaxAxisDeriv(op: ArgMaxAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type ArgMinAxisDeriv(op: ArgMinAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type GatherDeriv(op: Gather) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type ScatterDeriv(op: Scatter) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary -dOp // TODO


[<OpExtender>]
type StoreDeriv(op: Store) =
    interface IDerivableOp with       
        member this.Deriv dOp = Map.empty


[<OpExtender>]
type AssumeZeroDerivDeriv(op: AssumeZeroDeriv) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary dOp // TODO
    

[<OpExtender>]
type AssumeDerivDeriv(op: AssumeDeriv) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary dOp // TODO
    

[<OpExtender>]
type AnnotatedDeriv(op: Annotated) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary dOp 
 

[<OpExtender>]
type PrintDeriv(op: Print) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary dOp 
    

[<OpExtender>]
type DumpDeriv(op: Dump) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary dOp 


[<OpExtender>]
type CheckFiniteDeriv(op: CheckFinite) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary dOp 


[<OpExtender>]
type ChannelDeriv(op: Channel) =
    interface IDerivableOp with      
        member this.Deriv dOp = Args.unary dOp 


