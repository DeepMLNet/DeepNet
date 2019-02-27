namespace Tensor.Expr.DerivOps

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops


[<OpExtender>]
type UnaryPlusDeriv(op: UnaryPlus) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let env = DerivTools.Env.make op dOp
            env.DOp |> DerivTools.unary    


[<OpExtender>]
type NegateDeriv(op: Negate) =
    interface IDerivableOp with
        member this.Deriv dOp =  
            let env = DerivTools.Env.make op dOp
            -env.DOp |> DerivTools.unary               


[<OpExtender>]
type AbsDeriv(op: Abs) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let env = DerivTools.Env.make op dOp
            env.DOp * Expr.padLeft (Expr.signt env.Only) |> DerivTools.unary 


[<OpExtender>]
type SignTDeriv(op: SignT) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            let env = DerivTools.Env.make op dOp
            env.Zeros env.Only |> DerivTools.unary 
            

[<OpExtender>]
type LogDeriv(op: Log) =
    interface IDerivableOp with
        member this.Deriv dOp =  
            let env = DerivTools.Env.make op dOp
            env.DOp * Expr.padLeft (env.Only ** (-env.One)) |> DerivTools.unary
      

[<OpExtender>]
type Log10Deriv(op: Log10) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (env.Only ** (-env.One) / log env.Ten) |> DerivTools.unary


[<OpExtender>]
type ExpDeriv(op: Exp) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (exp env.Only) |> DerivTools.unary


[<OpExtender>]
type SinDeriv(op: Sin) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (cos env.Only) |> DerivTools.unary


[<OpExtender>]
type CosDeriv(op: Cos) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (-sin env.Only) |> DerivTools.unary

        
[<OpExtender>]
type TanDeriv(op: Tan) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (env.One + (tan env.Only) ** env.Two) |> DerivTools.unary


[<OpExtender>]
type AsinDeriv(op: Asin) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (env.One / Expr.sqrtt (env.One - env.Only ** env.Two)) |> DerivTools.unary


[<OpExtender>]
type AcosDeriv(op: Acos) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp
            env.DOp * Expr.padLeft (-env.One / Expr.sqrtt (env.One - env.Only ** env.Two)) |> DerivTools.unary


[<OpExtender>]
type AtanDeriv(op: Atan) =
    interface IDerivableOp with
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp
            env.DOp * Expr.padLeft (env.One / (env.One + env.Only ** env.Two)) |> DerivTools.unary 


[<OpExtender>]
type SinhDeriv(op: Sinh) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (cosh env.Only) |> DerivTools.unary


[<OpExtender>]
type CoshDeriv(op: Cosh) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (sinh env.Only) |> DerivTools.unary


[<OpExtender>]
type TanhDeriv(op: Tanh) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (env.One - (tanh env.Only) ** env.Two) |> DerivTools.unary
        

[<OpExtender>]
type SqrtDeriv(op: Sqrt) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp * Expr.padLeft (env.One / (env.Two * Expr.sqrtt env.Only)) |> DerivTools.unary


[<OpExtender>]
type CeilingDeriv(op: Ceiling) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.Zeros env.Only |> DerivTools.unary


[<OpExtender>]
type FloorDeriv(op: Floor) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.Zeros env.Only |> DerivTools.unary


[<OpExtender>]
type RoundDeriv(op: Round) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.Zeros env.Only |> DerivTools.unary


[<OpExtender>]
type TruncateDeriv(op: Truncate) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.Zeros env.Only |> DerivTools.unary


[<OpExtender>]
type InvertDeriv(op: Invert) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            -(Expr.padLeft env.Expr.T) .* env.DOp .* (Expr.padLeft env.Expr.T) |> DerivTools.unary 


[<OpExtender>]
type NotDeriv(op: Not) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.Zeros env.Only |> DerivTools.unary


[<OpExtender>]
type ReshapeDeriv(op: Reshape) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp
            env.DOp |> Expr.reshape (env.FunElems :: env.Only.Shape) |> DerivTools.unary


[<OpExtender>]
type DoBroadcastDeriv(op: DoBroadcast) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let mutable dOpUnBc = env.DOp
            for ax, (bSize, xSize) in List.indexed (List.zip env.Expr.Shape env.Only.Shape) do
                match bSize, xSize with
                | SizeSpec.Broadcast, SizeSpec.Broadcast -> ()
                | _, SizeSpec.Broadcast ->
                    dOpUnBc <- dOpUnBc |> Expr.sumKeepingAxis (ax + 1)
                | _ -> ()
            dOpUnBc |> DerivTools.unary                 


[<OpExtender>]
type PermuteAxesDeriv(op: PermuteAxes) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp
            let backPerm = Permutation.invert op.Permutation
            let dOpPerm = 
                0 :: List.map (fun p -> p + 1) backPerm
            env.DOp |> Expr.permuteAxes dOpPerm |> DerivTools.unary                 


[<OpExtender>]
type SubtensorDeriv(op: Subtensor) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let agExpanded = 
                Expr.zerosOfType env.DOp.DataType env.DOp.Dev (env.FunElems :: env.Only.Shape)
            env.DOp
            |> Expr.setSubtensor agExpanded.[SimpleRangeSpec.All :: op.Range] 
            |> DerivTools.unary


[<OpExtender>]
type ReverseAxisDeriv(op: ReverseAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp
            env.DOp |> Expr.reverseAxis (op.Axis + 1) |> DerivTools.unary


[<OpExtender>]
type DiagDeriv(op: Diag) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp |> Expr.diagMatAxis (op.Axis1 + 1) (op.Axis2 + 1) |> DerivTools.unary


[<OpExtender>]
type DiagMatDeriv(op: DiagMat) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp |> Expr.diagAxis (op.Axis1 + 1) (op.Axis2 + 1) |> DerivTools.unary
 
 
[<OpExtender>]
type SumAxisDeriv(op: SumAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let bcEgExp = env.DOp |> Expr.reshape (env.DOp.Shape |> ShapeSpec.insertBroadcastAxis (op.Axis + 1))
            bcEgExp |> Expr.broadcast (bcEgExp.Shape |> ShapeSpec.set (op.Axis + 1) env.Only.Shape.[op.Axis]) |> DerivTools.unary 


[<OpExtender>]
type ProductAxisDeriv(op: ProductAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            // TODO: This division method incorrectly returns NaN for zero elements.
            //       But currently I do not see any efficient alternative.
            let aBc = env.Only |> Expr.reshape (SizeSpec.broadcastable :: ShapeSpec.flatten env.Only.Shape)
            let pBc = env.Expr |> Expr.reshape [SizeSpec.broadcastable; SizeSpec.broadcastable]
            (env.DOpJac |> Expr.enableBroadcast 1) * (pBc / aBc) |> DerivTools.unary


[<OpExtender>]
type MaxAxisDeriv(op: MaxAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let bcExpr = env.Expr |> Expr.reshape (env.Expr.Shape |> ShapeSpec.insertBroadcastAxis op.Axis)
            let bcEgExp = env.DOp |> Expr.reshape (env.DOp.Shape |> ShapeSpec.insertBroadcastAxis (op.Axis + 1))
            Expr.ifThenElse (Expr.padLeft (env.Only ==== bcExpr)) bcEgExp (Expr.zerosLike bcEgExp) |> DerivTools.unary


[<OpExtender>]
type MinAxisDeriv(op: MinAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.Zeros env.Only |> DerivTools.unary


[<OpExtender>]
type ArgMaxAxisDeriv(op: ArgMaxAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.Zeros env.Only |> DerivTools.unary


[<OpExtender>]
type ArgMinAxisDeriv(op: ArgMinAxis) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.Zeros env.Only |> DerivTools.unary


[<OpExtender>]
type GatherDeriv(op: Gather) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dIndices = op.Indices |> List.map (Option.map (Expr >> Expr.padLeft))
            env.DOp |> Expr.scatter (None::dIndices) (env.FunElems :: env.Only.Shape) |> DerivTools.unary


[<OpExtender>]
type ScatterDeriv(op: Scatter) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dIndices = op.Indices |> List.map (Option.map (fun idx -> 
                idx |> Expr |> Expr.broadcastToShape (env.FunElems :: idx.Shape)))                   
            env.DOp |> Expr.gather (None::dIndices) |> DerivTools.unary


[<OpExtender>]
type StoreDeriv(op: Store) =
    interface IDerivableOp with       
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            Map.empty


[<OpExtender>]
type AssumeZeroDerivDeriv(op: AssumeZeroDeriv) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.Zeros env.Only |> DerivTools.unary
    

[<OpExtender>]
type AssumeDerivDeriv(op: AssumeDeriv) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            // TODO: does this op make sense the way it currently works?
            let env = DerivTools.Env.make op dOp 
            let deriv = Expr op.Deriv 
            match env.FunElems, deriv.Shape.[0] with
            | fl, jl when fl = jl -> deriv
            | fl, jl when jl = SizeSpec.broadcastable -> 
                deriv |> Expr.broadcast [fl; deriv.Shape.[1]]
            | _ -> failwithf "Cannot broadcast specified Jacobian of shape %A to required 
                              Jacobian shape %A" deriv.Shape env.DOp.Shape
            |> DerivTools.unary
    

[<OpExtender>]
type AnnotatedDeriv(op: Annotated) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp |> DerivTools.unary
 

[<OpExtender>]
type PrintDeriv(op: Print) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp |> DerivTools.unary
    

[<OpExtender>]
type DumpDeriv(op: Dump) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp |> DerivTools.unary


[<OpExtender>]
type CheckFiniteDeriv(op: CheckFinite) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            env.DOp |> Expr.checkFinite (sprintf "Derivative wrt %s" op.Label) |> DerivTools.unary 


[<OpExtender>]
type ChannelDeriv(op: Channel) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            failwith "TODO"


