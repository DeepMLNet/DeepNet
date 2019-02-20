namespace Tensor.Expr


module internal FracUtils = 
    /// greatest common divisor of a and b 
    let rec gcd a b =
        // Euclidean algorithm
        if a < 0L then gcd -a b
        elif b < 0L then gcd a -b
        elif a = 0L then b
        elif b = 0L then a
        elif a < b then gcd b a
        else
            //let q = a / b
            let r = a % b
            if r = 0L then b
            else gcd b r

    /// least common multiple of a and b
    let lcm a b =
        abs (a * b) / gcd a b


/// A rational number.
[<Struct; StructuredFormatDisplay("{Pretty}")>]
type Frac = 
    val Nom: int64
    val Dnm: int64

    new (nom, dnm) = 
        let nom, dnm =
            match dnm with
            | 0L -> failwith "denominator cannot be zero"
            | _ when dnm < 0L -> -nom, -dnm
            | _ -> nom, dnm
        let cd = FracUtils.gcd nom dnm
        {Nom=nom/cd; Dnm=dnm/cd}
    new (value) = Frac (value, 1L)               

    static member (~-) (a: Frac) = Frac (-a.Nom, a.Dnm)
    static member (+) (a: Frac, b: Frac) = Frac (a.Nom * b.Dnm + b.Nom * a.Dnm, a.Dnm * b.Dnm)
    static member (-) (a: Frac, b: Frac) = a + (-b)
    static member (*) (a: Frac, b: Frac) = Frac (a.Nom * b.Nom, a.Dnm * b.Dnm)
    static member (/) (a: Frac, b: Frac) = Frac (a.Nom * b.Dnm, a.Dnm * b.Nom)
    static member (.=) (a: Frac, b: Frac) = a = b
    static member (.<>) (a: Frac, b: Frac) = a <> b
    static member get_Zero () = Frac (0L)
    static member get_One () = Frac (1L)

    static member (+) (a: Frac, b: int64) = a + Frac b
    static member (-) (a: Frac, b: int64) = a - Frac b
    static member (*) (a: Frac, b: int64) = a * Frac b
    static member (/) (a: Frac, b: int64) = a / Frac b
    static member (.=) (a: Frac, b: int64) = a .= Frac b
    static member (.<>) (a: Frac, b: int64) = a .<> Frac b

    static member (+) (a: int64, b: Frac) = Frac a + b
    static member (-) (a: int64, b: Frac) = Frac a - b
    static member (*) (a: int64, b: Frac) = Frac a * b
    static member (/) (a: int64, b: Frac) = Frac a / b
    static member (.=) (a: int64, b: Frac) = Frac a .= b
    static member (.<>) (a: int64, b: Frac) = Frac a .<> b
         
    member this.IntValue = 
        if this.Dnm = 1L then this.Nom
        else failwithf "%A is not an integer" this

    member this.Pretty =
        if this.Dnm = 1L then sprintf "%d" this.Nom
        else sprintf "(%d/%d)" this.Nom this.Dnm

    static member nom (frac: Frac) = frac.Nom
    static member dnm (frac: Frac) = frac.Dnm
    static member ofInt i = Frac (i)
    static member toInt (frac: Frac) = frac.IntValue
    static member zero = Frac (0L)
    static member one = Frac (1L)

    static member roundTowardZero (f: Frac) = 
        Frac (f.Nom / f.Dnm)
    static member roundAwayFromZero (f: Frac) =
        if f.Nom % f.Dnm = 0L then
            Frac (f.Nom / f.Dnm)
        elif f.Nom > 0L then
            Frac (f.Nom / f.Dnm + 1L)
        else
            Frac (f.Nom / f.Dnm - 1L)

/// Active patterns for Frac.
module Frac =
    let (|Zero|_|) frac =
        if frac = Frac.zero then Some ()
        else None

    let (|One|_|) frac =
        if frac = Frac.one then Some ()
        else None

    let (|Integral|_|) (frac: Frac) =
        if frac.Dnm = 1L then Some frac.Nom
        else None

