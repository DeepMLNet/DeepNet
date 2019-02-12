namespace SymTensor.Deriv

open DeepNet.Utils
open SymTensor
open SymTensor.Ops


[<OpExtender>]
type UnaryPlusDeriv(op: UnaryPlus) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let env = Deriv.Env.make op dOp
            env.DOp |> Deriv.unary    


[<OpExtender>]
type NegateDeriv(op: Negate) =
    interface IDerivableOp with
        member this.Deriv dOp =  
            let env = Deriv.Env.make op dOp
            -env.DOp |> Deriv.unary               


[<OpExtender>]
type AbsDeriv(op: Abs) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let env = Deriv.Env.make op dOp
            env.DOp * Expr2.padLeft (Expr2.signt env.X) |> Deriv.unary 


[<OpExtender>]
type SignTDeriv(op: SignT) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let env = Deriv.Env.make op dOp
            env.Zeros env.X |> Deriv.unary 
            

[<OpExtender>]
type LogDeriv(op: Log) =
    interface IDerivableOp with
        member this.Deriv dOp =  
            let env = Deriv.Env.make op dOp
            env.DOp * Expr2.padLeft (env.X ** (-env.One)) |> Deriv.unary
      

[<OpExtender>]
type Log10Deriv(op: Log10) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (env.X ** (-env.One) / log env.Ten) |> Deriv.unary


[<OpExtender>]
type ExpDeriv(op: Exp) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (exp env.X) |> Deriv.unary


[<OpExtender>]
type SinDeriv(op: Sin) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (cos env.X) |> Deriv.unary


[<OpExtender>]
type CosDeriv(op: Cos) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (-sin env.X) |> Deriv.unary

        
[<OpExtender>]
type TanDeriv(op: Tan) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (env.One + (tan env.X) ** env.Two) |> Deriv.unary


[<OpExtender>]
type AsinDeriv(op: Asin) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (env.One / Expr2.sqrtt (env.One - env.X ** env.Two)) |> Deriv.unary


[<OpExtender>]
type AcosDeriv(op: Acos) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp
            env.DOp * Expr2.padLeft (-env.One / Expr2.sqrtt (env.One - env.X ** env.Two)) |> Deriv.unary


[<OpExtender>]
type AtanDeriv(op: Atan) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp
            env.DOp * Expr2.padLeft (env.One / (env.One + env.X ** env.Two)) |> Deriv.unary 


[<OpExtender>]
type SinhDeriv(op: Sinh) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (cosh env.X) |> Deriv.unary


[<OpExtender>]
type CoshDeriv(op: Cosh) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (sinh env.X) |> Deriv.unary


[<OpExtender>]
type TanhDeriv(op: Tanh) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (env.One - (tanh env.X) ** env.Two) |> Deriv.unary
        

[<OpExtender>]
type SqrtDeriv(op: Sqrt) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.DOp * Expr2.padLeft (env.One / (env.Two * Expr2.sqrtt env.X)) |> Deriv.unary


[<OpExtender>]
type CeilingDeriv(op: Ceiling) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.Zeros env.X |> Deriv.unary


[<OpExtender>]
type FloorDeriv(op: Floor) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.Zeros env.X |> Deriv.unary


[<OpExtender>]
type RoundDeriv(op: Round) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.Zeros env.X |> Deriv.unary


[<OpExtender>]
type TruncateDeriv(op: Truncate) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.Zeros env.X |> Deriv.unary


[<OpExtender>]
type InvertDeriv(op: Invert) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            -(Expr2.padLeft env.Expr.T) .* env.DOp .* (Expr2.padLeft env.Expr.T) |> Deriv.unary 


[<OpExtender>]
type NotDeriv(op: Not) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            env.Zeros env.X |> Deriv.unary


[<OpExtender>]
type ReshapeDeriv(op: Reshape) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp
            let funElems = dOp.Shape.[0]
            env.DOp |> Expr2.reshape (funElems :: env.X.Shape) |> Deriv.unary


[<OpExtender>]
type DoBroadcastDeriv(op: DoBroadcast) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            let mutable dOpUnBc = env.DOp
            for ax, (bSize, xSize) in List.indexed (List.zip env.Expr.Shape env.X.Shape) do
                match bSize, xSize with
                | SizeSpec.Broadcast, SizeSpec.Broadcast -> ()
                | _, SizeSpec.Broadcast ->
                    dOpUnBc <- dOpUnBc |> Expr2.sumKeepingAxis (ax + 1)
                | _ -> ()
            dOpUnBc |> Deriv.unary                 


[<OpExtender>]
type PermuteAxesDeriv(op: PermuteAxes) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp
            let backPerm = Permutation.invert op.Permutation
            let dOpPerm = 
                0 :: List.map (fun p -> p + 1) backPerm
            env.DOp |> Expr2.permuteAxes dOpPerm |> Deriv.unary                 


[<OpExtender>]
type SubtensorDeriv(op: Subtensor) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            let funElems = env.DOp.Shape.[0]
            let agExpanded = Expr2.zerosOfType dOp.DataType (funElems :: env.X.Shape)
            
            env.DOp
            |> Expr2.setSubtensor agExpanded.[SimpleRangeSpec.All :: op.Range] 
            |> Deriv.unary


[<OpExtender>]
type ReverseAxisDeriv(op: ReverseAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp
            env.DOp |> Expr2.reverseAxis (op.Axis + 1) |> Deriv.unary


[<OpExtender>]
type DiagDeriv(op: Diag) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"

[<OpExtender>]
type DiagMatDeriv(op: DiagMat) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type SumAxisDeriv(op: SumAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type ProductAxisDeriv(op: ProductAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type MaxAxisDeriv(op: MaxAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type MinAxisDeriv(op: MinAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type ArgMaxAxisDeriv(op: ArgMaxAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type ArgMinAxisDeriv(op: ArgMinAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type GatherDeriv(op: Gather) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type ScatterDeriv(op: Scatter) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type StoreDeriv(op: Store) =
    interface IDerivableOp with       
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            Map.empty


[<OpExtender>]
type AssumeZeroDerivDeriv(op: AssumeZeroDeriv) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"
    

[<OpExtender>]
type AssumeDerivDeriv(op: AssumeDeriv) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"
    

[<OpExtender>]
type AnnotatedDeriv(op: Annotated) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"
 

[<OpExtender>]
type PrintDeriv(op: Print) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"
    

[<OpExtender>]
type DumpDeriv(op: Dump) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type CheckFiniteDeriv(op: CheckFinite) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


[<OpExtender>]
type ChannelDeriv(op: Channel) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            failwith "TODO"


