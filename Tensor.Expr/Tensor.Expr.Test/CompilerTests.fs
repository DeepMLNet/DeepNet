namespace global
#nowarn "25"

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Expr.Base
open Tensor.Expr.Compiler
open TestUtils


type CompilerTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format
    
    [<Fact>]
    let ``Simple ExecutionRecipe`` () =
        let ctx = Context.root HostTensor.Dev
        let a = Var<float> (ctx / "a", [Size.fix 2L; Size.fix 3L])
        let b = Var<float> (ctx / "b", [Size.fix 2L; Size.fix 3L])        

        let expr = Expr Vars.a + Expr Vars.b
        
        let env : CompileEnv = {
            ExternalTargets = Set.empty
            DumpPath = Some "CompilerTest/SimpleExecutionRecipe.txt"
        }
        let group = BaseExprGroup [expr.BaseExpr]
        let er = ExecutionRecipe.make env group
        
        printfn "ExecutionRecipe:\n%A" er
        
        
        
        
