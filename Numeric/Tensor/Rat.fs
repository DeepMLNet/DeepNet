namespace Tensor

open System
open System.Numerics


/// A rational number, i.e. a fraction of arbitrary precision.
[<Struct; CustomEquality; CustomComparison; StructuredFormatDisplay("{Pretty}")>]
type Rat =
    
    // The default constructor for structs in .NET initializes all fields to zero and
    // this behavior cannot be changed. Thus we store the denominator minus one
    // to have "0/1" as our default value instead of "0/0" which would be NaN.

    // Rules for encoding non-finite numbers:
    // Num > 0 and Dnm = 0 => +Inf
    // Num < 0 and Dnm = 0 => -Inf
    // Num = 0 and Dnm = 0 => NaN

    /// the numerator
    val Num: bigint 
    /// the denominator minus one
    val private DnmMinusOne: bigint

    /// the denominator
    member a.Dnm = a.DnmMinusOne + bigint.One

    /// Constructs a fraction from numerator and denominator.
    new (num, dnm) = 
        if dnm <> bigint.Zero then
            // not infinity or NaN
            let num, dnm =
                if dnm < bigint.Zero then -num, -dnm
                else num, dnm
            let cd = bigint.GreatestCommonDivisor(num, dnm)
            {Num=num/cd; DnmMinusOne=dnm/cd - bigint.One}
        else
            // infinity or NaN
            match num with
            | _ when num > bigint.Zero -> {Num=bigint.One;      DnmMinusOne=bigint.MinusOne}
            | _ when num < bigint.Zero -> {Num=bigint.MinusOne; DnmMinusOne=bigint.MinusOne}
            | _ ->                        {Num=bigint.Zero;     DnmMinusOne=bigint.MinusOne}

    /// Constructs a fraction from numerator and denominator.
    new (num: int32, dnm: int32) = Rat (bigint num, bigint dnm)

    /// Constructs a fraction from numerator and denominator.
    new (num: uint32, dnm: uint32) = Rat (bigint num, bigint dnm)

    /// Constructs a fraction from numerator and denominator.
    new (num: int64, dnm: int64) = Rat (bigint num, bigint dnm)

    /// Constructs a fraction from numerator and denominator.
    new (num: uint64, dnm: uint64) = Rat (bigint num, bigint dnm)

    /// Constructs an integer rational number.
    new (value) = Rat (value, bigint.One)    

    /// Constructs an integer rational number.
    new (value: int32) = Rat (bigint value)   
    
    /// Constructs an integer rational number.
    new (value: uint32) = Rat (bigint value)    

    /// Constructs an integer rational number.
    new (value: int64) = Rat (bigint value)   
    
    /// Constructs an integer rational number.
    new (value: uint64) = Rat (bigint value)    

    /// True if this is an integer, i.e. denominator is one.
    member a.IsInt = a.Dnm = bigint.One

    /// True is this is positive infinity.
    member a.IsPosInf = 
        a.Num > bigint.Zero && a.Dnm = bigint.Zero

    /// True is this is negative infinity.
    member a.IsNegInf = 
        a.Num < bigint.Zero && a.Dnm = bigint.Zero

    /// True is this is infinity.
    member a.IsInf =
        a.IsPosInf || a.IsNegInf

    /// True if this is not-a-number.
    member a.IsNaN =
        a.Num = bigint.Zero && a.Dnm = bigint.Zero

    /// True if this is a finite number (not infinity and not NaN).
    member a.IsFinite =
        a.Dnm <> bigint.Zero

    /// Fails if this is not an integer rational.
    member private a.CheckInt() =
        if not a.IsInt then
            failwithf "the rational %A is not an integer" a

    // unary and binary operators
    static member (~+) (a: Rat) = a
    static member (~-) (a: Rat) = Rat (-a.Num, a.Dnm)
    static member Abs (a: Rat) = Rat (abs a.Num, a.Dnm)
    static member Floor (a: Rat) = 
        if a.Num >= bigint.Zero then
            Rat(a.Num - (a.Num % a.Dnm), a.Dnm)
        else
            let r = a.Num % a.Dnm // is negative
            if r <> bigint.Zero then Rat(a.Num - a.Dnm - r, a.Dnm)
            else a            
    static member Ceiling (a: Rat) = 
        if a.Num >= bigint.Zero then
            let r = a.Num % a.Dnm
            if r <> bigint.Zero then Rat(a.Num + (a.Dnm - r), a.Dnm)
            else a
        else
            Rat(a.Num - (a.Num % a.Dnm), a.Dnm)
    static member Truncate (a: Rat) =
        Rat(a.Num - (a.Num % a.Dnm), a.Dnm)
    static member (+) (a: Rat, b: Rat) = 
        if a.IsNaN || b.IsNaN then Rat.NaN
        else
            match a.IsNegInf, a.IsPosInf, b.IsNegInf, b.IsPosInf with
            |           true,          _,          _,       true -> Rat.NaN
            |              _,       true,       true,          _ -> Rat.NaN
            |           true,          _,          _,          _ -> Rat.NegInf
            |              _,          _,       true,          _ -> Rat.NegInf
            |              _,       true,          _,          _ -> Rat.PosInf
            |              _,          _,          _,       true -> Rat.PosInf
            |          false,      false,      false,      false ->
                Rat (a.Num * b.Dnm + b.Num * a.Dnm, a.Dnm * b.Dnm)
    static member (-) (a: Rat, b: Rat) = a + (-b)
    static member (*) (a: Rat, b: Rat) = 
        Rat (a.Num * b.Num, a.Dnm * b.Dnm)
    static member (/) (a: Rat, b: Rat) = 
        Rat (a.Num * b.Dnm, a.Dnm * b.Num)
    static member (%) (a: Rat, b: Rat) = 
        Rat ((a.Num * b.Dnm) % (b.Num * a.Dnm), a.Dnm * b.Dnm)
    static member get_Sign (a: Rat) = sign a.Num
    static member Zero = Rat (bigint.Zero)
    static member One = Rat (bigint.One)
    static member MinusOne = Rat (bigint.MinusOne)
    static member PosInf = Rat (bigint.One, bigint.Zero)
    static member NegInf = Rat (bigint.MinusOne, bigint.Zero)
    static member NaN = Rat (bigint.Zero, bigint.Zero)
    static member MaxValue = Rat.PosInf
    static member MinValue = Rat.NegInf
    
    // conversions to other types
    static member op_Explicit(a: Rat) : bigint = a.CheckInt(); a.Num
    static member op_Explicit(a: Rat) : int32 = a.CheckInt(); int32 a.Num
    static member op_Explicit(a: Rat) : uint32 = a.CheckInt(); uint32 a.Num
    static member op_Explicit(a: Rat) : int64 = a.CheckInt(); int64 a.Num
    static member op_Explicit(a: Rat) : uint64 = a.CheckInt(); uint64 a.Num
    member private a.ToDouble () =
        if a.IsPosInf then System.Double.PositiveInfinity
        elif a.IsNegInf then System.Double.NegativeInfinity
        elif a.IsNaN then System.Double.NaN
        else double a.Num / double a.Dnm
    static member op_Explicit(a: Rat) : double = a.ToDouble()              
    static member op_Explicit(a: Rat) : single = a.ToDouble() |> single

    // conversions from other types
    static member op_Implicit(v: bigint) = Rat (v)
    static member op_Implicit(v: int32) = Rat (v)
    static member op_Implicit(v: uint32) = Rat (v)
    static member op_Implicit(v: int64) = Rat (v)
    static member op_Implicit(v: uint64) = Rat (v)


    // equation and comparison
    // NaN never equals NaN, but comparing NaN to NaN puts them in the same sort place.
    interface IEquatable<Rat> with
        member a.Equals b = 
            if a.IsNaN || b.IsNaN then false
            else a.Num = b.Num && a.Dnm = b.Dnm
    override a.Equals b =
        match b with
        | :? Rat as b -> (a :> IEquatable<_>).Equals b
        | _ -> failwith "can only equate to another Rat"

    interface IComparable<Rat> with
        member a.CompareTo b = 
            if   a.IsNaN && b.IsNaN then 0
            elif a.IsNaN then -1
            elif b.IsNaN then 1
            elif a.IsNegInf && b.IsNegInf then 0
            elif a.IsNegInf then -1
            elif b.IsNegInf then 1
            elif a.IsPosInf && b.IsPosInf then 0
            elif a.IsPosInf then 1
            elif b.IsPosInf then -1
            else compare (a.Num * b.Dnm) (b.Num * a.Dnm)
    interface IComparable with
        member a.CompareTo b =
            match b with
            | :? Rat as b -> (a :> IComparable<Rat>).CompareTo b
            | _ -> failwith "can only compare to another Rat"

    override a.GetHashCode() = hash (a.Num, a.Dnm)

    /// Pretty string representation.
    member a.Pretty = 
        if   a.IsNaN    then "NaN"
        elif a.IsNegInf then "-Inf"
        elif a.IsPosInf then "Inf"
        elif a.IsInt    then sprintf "%A" a.Num
        else sprintf "%A/%A" a.Num a.Dnm
    override a.ToString() = a.Pretty


/// A rational number, i.e. a fraction of arbitrary precision.
module Rat =
    /// numerator
    let num (a: Rat) = a.Num
    /// denominator
    let dnm (a: Rat) = a.Dnm
    /// True if a is an integer, i.e. its denominator is one.
    let isInteger (a: Rat) = a.IsInt
    /// True if a is positive infinity.
    let isPosInf (a: Rat) = a.IsPosInf
    /// True if a is negative infinity.
    let isNegInf (a: Rat) = a.IsNegInf
    /// True if a is infinity.
    let isInf (a: Rat) = a.IsInf
    /// True if a is not-a-number.
    let isNaN (a: Rat) = a.IsNaN
    /// True if a is a finite number.
    let isFinite (a: Rat) = a.IsFinite


/// Active recognizers for rational numbers.
[<AutoOpen>]
module RatRecognizers =
    /// decomposes a rational into its numerator and denominator
    let (|Rat|) (a: Rat) = (a.Num, a.Dnm)

    /// RatFrac(num,dnm) matches a true (finite) fraction and 
    /// RatInteger(value) matches an integer rational number.
    let (|RatFrac|RatInt|RatPosInf|RatNegInf|RatNaN|) (a: Rat) =
        if a.IsNaN then RatNaN
        elif a.IsPosInf then RatPosInf
        elif a.IsNegInf then RatNegInf
        elif a.IsInt then RatInt a.Num
        else RatFrac (a.Num, a.Dnm)

