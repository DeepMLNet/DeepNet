#r "../Tensor/bin/Debug/ManagedCuda.dll"
#r "../Tensor/bin/Debug/Tensor.dll"


open Tensor

let a = Rat (1, 2)
let b = Rat (1, 2)
let c = Rat (2, 2)
let d = Rat (12, 8)


match d with
| Rat(da, db) -> printfn "%A %A" da db

let myFunc (Rat(da, db)) =
    printfn "%A %A" da db



a = b
a > b
c = a
c > a
c >= c

-c

float c

let b1 = bigint 100
let b2 = bigint 103

b2 > b1


sprintf "%A" b1


bigint 3.3


float b1

let bbig = bigint 100000 * bigint 100000
int bbig


let fl = 2.3
bigint.GreatestCommonDivisor(bigint 0, bigint 0)

uint32 -3

max 2 3


System.Double.PositiveInfinity = System.Double.PositiveInfinity
System.Double.PositiveInfinity > 3.0
///ist.sum

System.Double.NaN = System.Double.NaN
System.Double.NaN <> System.Double.NaN
System.Double.NaN > 3.0

compare System.Double.NaN System.Double.NaN
compare System.Double.NaN System.Double.NegativeInfinity

compare 3.0 System.Double.NaN


compare System.Double.NegativeInfinity System.Double.NaN


type MyType =
    static member Zero = 111

typeof<MyType>.GetProperty("Zero", typeof<int>).GetValue(null)

