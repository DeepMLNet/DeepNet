namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open Tensor.Utils
open Tensor
open Tensor.Algorithm


type BigIntegerTests (output: ITestOutputHelper) =

    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let testBezout x y =
        let gcd, a, b = bigint.Bezout (x, y)
        let gcd2 = bigint.GreatestCommonDivisor(x, y)
        let gcd3 = a * x + b * y       
        gcd2 |> should equal gcd
        gcd3 |> should equal gcd    
        printfn "gcd(%A, %A) = %A * %A + %A * %A = %A" x y a x b y gcd

    [<Fact>]
    let Bezout () =
        testBezout (bigint (2 * 3 * 3 * 5 * 11)) (bigint (2 * 3 * 5 * 13))
        testBezout (bigint (-2 * 3 * 3 * 5 * 11)) (bigint (2 * 3 * 5 * 13))
        testBezout (bigint (2 * 3 * 3 * 5 * 11)) (-bigint (2 * 3 * 5 * 13))
        testBezout (-bigint (2 * 3 * 3 * 5 * 11)) (-bigint (2 * 3 * 5 * 13))
        
    [<Fact>]
    let ``Bezout Zero`` () =
        testBezout (bigint 0) (bigint (2 * 3 * 5 * 13))
        testBezout (bigint (2 * 3)) (bigint 0)
        testBezout (bigint 0) (bigint 0)

    [<Fact>]
    let ``Bezout Random`` () =
        let rnd = System.Random 123
        for i=1 to 100 do
            let x = rnd.Next() |> bigint
            let y = rnd.Next() |> bigint 
            testBezout x y
            