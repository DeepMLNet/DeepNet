open System


let rec gcd a b =
    // Euclidean algorithm
    if a < b then gcd b a
    elif a = 0 then b
    elif b = 0 then a
    else
        printfn "a: %d  b: %d" a b
        let q = a / b
        let r = a % b
        printfn "q: %d  r: %d" q r
        if r = 0 then b
        else gcd b r


[<StructuredFormatDisplay("{Pretty}")>]
[<Struct>]
/// a rational number
type FractionT = 
    val Nom: int
    val Dnm: int

    new (nom, dnm) = 
        let nom, dnm =
            match dnm with
            | 0 -> failwith "denominator cannot be zero"
            | _ when dnm < 0 -> -nom, -dnm
            | _ -> nom, dnm
        let cd = gcd nom dnm
        {Nom=nom/cd; Dnm=dnm/cd}
    new (value) = FractionT (value, 1)               

    static member (~-) (a: FractionT) = FractionT (-a.Nom, a.Dnm)
    static member (+) (a: FractionT, b: FractionT) = FractionT (a.Nom * b.Dnm + b.Nom * a.Dnm, a.Dnm * b.Dnm)
    static member (-) (a: FractionT, b: FractionT) = a + (-b)
    static member (*) (a: FractionT, b: FractionT) = FractionT (a.Nom * b.Nom, a.Dnm * b.Dnm)
    static member (/) (a: FractionT, b: FractionT) = FractionT (a.Nom * b.Dnm, a.Dnm * b.Nom)

    static member (+) (a: FractionT, b: int) = a + FractionT b
    static member (-) (a: FractionT, b: int) = a - FractionT b
    static member (*) (a: FractionT, b: int) = a * FractionT b
    static member (/) (a: FractionT, b: int) = a / FractionT b
    static member (==) (a: FractionT, b: int) = a = FractionT b

    static member (+) (a: int, b: FractionT) = FractionT a + b
    static member (-) (a: int, b: FractionT) = FractionT a - b
    static member (*) (a: int, b: FractionT) = FractionT a * b
    static member (/) (a: int, b: FractionT) = FractionT a / b
         
    member this.Pretty =
        if this.Dnm = 1 then sprintf "%d" this.Nom
        else sprintf "(%d/%d)" this.Nom this.Dnm


// testing
let a = FractionT 1
let b = a/ 2
let c = b * 2
let d= FractionT 0

a = c
a <> c
a <> b 
//let b = a / 2

a == 1




