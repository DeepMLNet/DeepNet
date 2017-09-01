namespace Tensor

open System
open System.Numerics


/// A rational number, i.e. a fraction of arbitrary precision.
[<Struct; CustomEquality; CustomComparison; StructuredFormatDisplay("{Pretty}")>]
type Rat =
    
    // The default constructor in .NET initializes all fields to zero and
    // this behavior cannot be changed. Thus we store the denominator minus one
    // to have "0/1" as our default value instead of "0/0" which would be
    // undefined.

    /// the numerator
    val Num: bigint 
    /// the denominator minus one
    val private DnmMinusOne: bigint

    /// the denominator
    member a.Dnm = a.DnmMinusOne + bigint.One

    /// Constructs a fraction from numerator and denominator.
    new (num, dnm) = 
        let num, dnm =
            match dnm with
            | _ when dnm = bigint.Zero -> 
                raise (DivideByZeroException("denominator cannot be zero"))
            | _ when dnm < bigint.Zero -> -num, -dnm
            | _ -> num, dnm
        let cd = bigint.GreatestCommonDivisor(num, dnm)
        {Num=num/cd; DnmMinusOne=dnm/cd - bigint.One}

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
    member a.IsInteger = a.Dnm = bigint.One

    /// Fails if this is not an integer rational.
    member private a.CheckInteger() =
        if not a.IsInteger then
            failwithf "the rational %A is not an integer" a

    // unary and binary operators
    static member (~+) (a: Rat) = a
    static member (~-) (a: Rat) = Rat (-a.Num, a.Dnm)
    static member Abs (a: Rat) = Rat (abs a.Num, a.Dnm)
    static member (+) (a: Rat, b: Rat) = 
        Rat (a.Num * b.Dnm + b.Num * a.Dnm, a.Dnm * b.Dnm)
    static member (-) (a: Rat, b: Rat) = 
        Rat (a.Num * b.Dnm - b.Num * a.Dnm, a.Dnm * b.Dnm)
    static member (*) (a: Rat, b: Rat) = Rat (a.Num * b.Num, a.Dnm * b.Dnm)
    static member (/) (a: Rat, b: Rat) = Rat (a.Num * b.Dnm, a.Dnm * b.Num)
    static member (%) (a: Rat, b: Rat) = 
        Rat ((a.Num * b.Dnm) % (b.Num * a.Dnm), a.Dnm * b.Dnm)
    static member get_Sign (a: Rat) = sign a.Num
    static member Zero = Rat (bigint.Zero)
    static member One = Rat (bigint.One)
    static member MinusOne = Rat (bigint.MinusOne)

    // conversions
    static member op_Explicit(a: Rat) : int32 = a.CheckInteger(); int32 a.Num
    static member op_Explicit(a: Rat) : uint32 = a.CheckInteger(); uint32 a.Num
    static member op_Explicit(a: Rat) : int64 = a.CheckInteger(); int64 a.Num
    static member op_Explicit(a: Rat) : uint64 = a.CheckInteger(); uint64 a.Num
    static member op_Explicit(a: Rat) : single = single a.Num / single a.Dnm
    static member op_Explicit(a: Rat) : double = double a.Num / double a.Dnm

    interface IEquatable<Rat> with
        member a.Equals b = a.Num = b.Num && a.Dnm = b.Dnm
    override a.Equals b =
        match b with
        | :? Rat as b -> (a :> IEquatable<_>).Equals b
        | _ -> failwith "can only equate to another Rat"

    interface IComparable<Rat> with
        member a.CompareTo b = compare (a.Num * b.Dnm) (b.Num * a.Dnm)
    interface IComparable with
        member a.CompareTo b =
            match b with
            | :? Rat as b -> (a :> IComparable<Rat>).CompareTo b
            | _ -> failwith "can only compare to another Rat"

    override a.GetHashCode() = hash (a.Num, a.Dnm)

    /// Pretty string representation.
    member a.Pretty = sprintf "%A/%A" a.Num a.Dnm
    override a.ToString() = a.Pretty


/// A rational number, i.e. a fraction of arbitrary precision.
module Rat =
    /// numerator
    let num (a: Rat) = a.Num
    /// denominator
    let dnm (a: Rat) = a.Dnm
    /// True if a is an integer, i.e. its denominator is one.
    let isInteger (a: Rat) = a.IsInteger


/// Active recognizers for rational numbers.
[<AutoOpen>]
module RatRecognizers =
    /// decomposes a rational into its numerator and denominator
    let (|Rat|) (a: Rat) = (a.Num, a.Dnm)

    /// RatFrac(num,dnm) matches a true fraction and 
    /// RatInteger(value) matches an integer rational number.
    let (|RatFrac|RatInteger|) (a: Rat) =
        if a.IsInteger then RatInteger a.Num
        else RatFrac (a.Num, a.Dnm)

