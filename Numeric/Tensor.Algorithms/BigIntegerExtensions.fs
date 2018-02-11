namespace Tensor.Algorithms

open System.Numerics

/// Extensions to System.Numerics.BigInteger.
[<AutoOpen>]
module BigIntegerExtensions =

    type System.Numerics.BigInteger with
    
        /// Computes the GCD of a and b and the coefficients of Bezout's identity using 
        /// the extended Euclidean algorithm.
        /// Returns a tuple of (gcd, x, y) so that: a*x + b*y = gcd(a,b).
        /// The returned GCD is always non-negative and gcd(0, 0)=0.
        static member Bezout (a: bigint, b: bigint) =       
            let rec step (rp: bigint) (r: bigint) (sp: bigint) (s: bigint) (tp: bigint) (t: bigint) =
                // We implement the algorithm as described in https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
                // using the following variable names:
                // q = q_i
                // rp = r_{i-1}; r = r_i; rn = r_{i+1}
                // sp = s_{i-1}; s = s_i; rn = s_{i+1} 
                // tp = t_{i-1}; t = t_i; rn = t_{i+1}
                if r.IsZero then rp, sp, tp
                else                 
                    let q = rp / r
                    let rn = rp - q * r
                    let sn = sp - q * s
                    let tn = tp - q * t
                    step r rn s sn t tn
    
            let gcd, xa, ya = step (abs a) (abs b) bigint.One bigint.Zero bigint.Zero bigint.One
            let x = if a < bigint 0 then -xa else xa
            let y = if b < bigint 0 then -ya else ya
            gcd, x, y        
            
        