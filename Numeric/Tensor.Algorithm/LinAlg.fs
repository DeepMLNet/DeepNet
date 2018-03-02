namespace Tensor.Algorithm

open Tensor
open Tensor.Utils


/// Linear algebra algorithms
module LinAlg =

    let private getRowsCols (L: Tensor<_>) =
        match L.Shape with
        | [rows; cols] -> rows, cols
        | _ -> failwithf "L must be a matrix but it has shape %A" L.Shape

    /// Computes the reduced row echelon form of matrix L augmented with matrix A.
    /// Returns a tuple of 
    /// (row echelon form, number of non-zero rows, list of non-normalized columns,
    ///  transformed augmentation matrix).
    let rowEchelonAugmented (L: Tensor<'T>) (A: Tensor<'T>) =
        let rows, cols = getRowsCols L
        match A.Shape with
        | [ar; _] when ar = rows -> ()
        | _ -> 
            failwithf "augmentation A (%A) must be a matrix with same number of rows as L (%A)"
                      A.Shape L.Shape
        let R, B = Tensor.copy L, Tensor.copy A

        let swapRows (M: Tensor<_>) i j =
            let tmp = M.[i, *] |> Tensor.copy
            M.[i, *] <- M.[j, *]
            M.[j, *] <- tmp

        // step of Gaussian Elimination algroithm
        let rec step unnormalizedCols row col =
            if row < rows && col < cols then
                // find pivot row by maximum magnitude
                let pivot = 
                    R.[row.., col] 
                    |> abs 
                    |> Tensor.argMax 
                    |> List.exactlyOne
                    |> fun p -> p + row

                // swap active row with pivot row
                swapRows R row pivot
                swapRows B row pivot

                let pivotVal = R.[row, col] |> Tensor.copy
                if Tensor.value pivotVal <> zero<'T> then
                    // make active row start with a one
                    R.[row, *] <- R.[row, *] / pivotVal
                    B.[row, *] <- B.[row, *] / pivotVal   

                    // eliminate active column from all other rows
                    let facs = R.[0L .. row-1L, col..col] |> Tensor.copy
                    R.[0L .. row-1L, *] <- R.[0L .. row-1L, *] - facs * R.[row..row, *]
                    B.[0L .. row-1L, *] <- B.[0L .. row-1L, *] - facs * B.[row..row, *]
                    let facs = R.[row+1L .., col..col] |> Tensor.copy
                    R.[row+1L .., *] <- R.[row+1L .., *] - facs * R.[row..row, *]
                    B.[row+1L .., *] <- B.[row+1L .., *] - facs * B.[row..row, *]

                    // continue with next row and column
                    step unnormalizedCols (row+1L) (col+1L)
                else
                    // All remaining entries in active column are zero,
                    // thus it cannot be normalized.
                    step (col::unnormalizedCols) row (col+1L)
            elif col < cols then
                // remaining columns cannot be normalized
                step (col::unnormalizedCols) row (col+1L)
            else
                // remaining rows must be zero
                row, List.rev unnormalizedCols

        let nonZeroRows, unnormalizedCols = step [] 0L 0L
        R, nonZeroRows, unnormalizedCols, B

    /// Computes the reduced row echelon form of the specified matrix.
    /// Returns a tuple of 
    /// (row echelon form, number of non-zero rows, list of non-normalized columns).
    let rowEchelon (L: Tensor<'T>) =
        let rows, _ = getRowsCols L
        let A = Tensor.zeros L.Dev [rows; 0L]
        let R, nonZeroRows, unnormalizedCols, _ = rowEchelonAugmented L A
        R, nonZeroRows, unnormalizedCols

    /// Computes the generalized inverse I, solvability constraints S 
    /// and null-space N of the specified matrix L.
    /// L can be of any shape and rank.
    /// The return values is a tuple (I, S, N).
    /// The following properties are fulfilled.
    /// Inverse:     M .* I .* M = M.
    /// Solvability: S .* M = 0.
    /// Null-space:  M .* N = 0.
    /// This has the following consequences for a linear equation system of the
    /// form y = M .* x:
    /// If y comes from the solvable space (i.e. S .* y = 0), 
    /// then the value x = I .* y gives one solution of y = M .* x.
    /// Adding any linear combination of the columns of N to this x yields 
    /// another solution, i.e. y = M .* x = M .* (x + N .* z) for any z.
    /// If x contains no component from the null-space (i.e. N^T .* x = 0),
    /// then we can recover x from y = M .* x by x = I .* y.
    /// (TODO: check last sentence, because it was violated in test1)
    let generalInverse (L: Tensor<'T>) =
        // compute row echelon form
        let rows, cols = getRowsCols L
        let A = HostTensor.identity (max rows cols) |> fun A -> A.[0L .. rows-1L, *]       
        let E, nzRows, unCols, I = rowEchelonAugmented L A

        // extract unnormalized columns from row echelon form E
        let U = 
            if List.isEmpty unCols then Tensor<'T>.zeros L.Dev [rows; 0L]
            else
                unCols
                |> List.map (fun c -> E.[*, c..c])
                |> Tensor.concat 1

        // zero unnormalized columns in row echelon form E
        // (setting undetermined values of solution to zero)
        for c in unCols do
            E.[*, c] <- Tensor.scalar E.Dev zero<'T>
        
        // calculate inverse of image
        let LI = (E.T .* I).[*, 0L .. rows-1L]

        // null space of L
        let N = -E.T .* U
        for c, r in List.indexed unCols do
            N.[[r; int64 c]] <- one<'T>

        // solvability constraint
        let S = I.[nzRows.., 0L .. rows-1L]

        LI, S, N


    /// <summary>Computes the Smith Normal Form S of integer matrix A and returns a tuple
    /// (U, S, V) so that S = U .* A .* V, where U and V are invertible matrices and
    /// S is a positive, diagonal matrix with the property that each element of the diagonal
    /// divides all of its successors.</summary> 
    /// <remarks>The Smith Normal Form exists for a matrix of any shape or rank.</remarks>
    let smithNormalForm (A: Tensor<bigint>) =       
             
        /// Swaps row1 with row2 in M.
        let swapRows row1 row2 (M: Tensor<bigint>) =
            let tmp = Tensor.copy M.[row1, *]
            M.[row1, *] <- M.[row2, *]
            M.[row2, *] <- tmp
        
        /// Replaces     row1 by f11 * row1 + f12 * row2
        /// and replaces row2 by f21 * row1 + f22 * row2 in M.    
        let linComb row1 row2 (f11, f12) (f21, f22) (M: Tensor<bigint>) =
            let v1, v2 = Tensor.copy M.[row1, *], Tensor.copy M.[row2, *]
            M.[row1, *] <- f11 * v1 + f12 * v2
            M.[row2, *] <- f21 * v1 + f22 * v2
                     
        /// Moves a non-zero column to the left and swaps rows so that the 
        /// element in the upper-left corner of the returned matrix is non-zero.
        /// M is modified in-place. Returns None if M is zero.   
        let movePivot (R: Tensor<bigint>) (C: Tensor<bigint>) (M: Tensor<bigint>) =
            // find left-most column with at least a non-zero entry
            match M <<>> bigint.Zero |> Tensor.anyAxis 0 |> Tensor.tryFind true with
            | Some [nzCol] ->
                // move non-zero column to the left
                if nzCol <> 0L then
                    swapRows 0L nzCol M.T
                    swapRows 0L nzCol C.T               
                // find first row that has non-zero element in left-most column
                let nzRow = M.[*, 0L] <<>> bigint.Zero |> Tensor.find true |> List.exactlyOne
                // and swap that row with the first row, if necessary
                if nzRow <> 0L then
                    swapRows 0L nzRow M
                    swapRows 0L nzRow R
                // non-zero column found                    
                true
            | None -> 
                // all columns are zero
                false
            | _ -> failwith "unexpected find result"                              

        /// Ensures that the pivot element in the upper-left corner of M divides the 
        /// first element of all following rows.
        /// This is done by replacing rows by linear combinations of other rows.
        let rec makePivotGCD (R: Tensor<bigint>) (M: Tensor<bigint>) changed =
            let pivot = M.[[0L; 0L]]           
            // find any row that is not divisable by pivot
            match (M.[*, 0L] % pivot) <<>> bigint.Zero |> Tensor.tryFind true with
            | Some [ndRow] ->
                // row ndRow is not divisble by pivot, because offender % pivot <> 0
                let offender = M.[[ndRow; 0L]]               
                // apply extended Euclidean algorithm to obtain:
                // beta = gcd(pivot, offender) = sigma * pivot + tau * offener
                let beta, sigma, tau = bigint.Bezout (pivot, offender)
                let gamma, alpha = offender / beta, pivot / beta                
                // replace rows by linear combinations so that after replacement:
                // 1. M.[[0L; 0L]] = beta = gcd(pivot, offender)
                // 2. M.[[ndRow; 0L]] = 0                
                linComb 0L ndRow (sigma, tau) (-gamma, alpha) M
                linComb 0L ndRow (sigma, tau) (-gamma, alpha) R            
                // iterate until all rows are divisible by pivot
                makePivotGCD R M true
            | None ->
                // all rows are divisable by pivot
                changed
            | _ -> failwith "unexpected find result"
        
        /// Ensures that the first element of all but the first row is zero by substracting
        /// an appropriate multiple of the first row from these rows.
        let eliminateRows (R: Tensor<bigint>) (M: Tensor<bigint>) =
            let f = M.[1L.., 0L..0L] / M.[0L, 0L]
            M.[1L.., *] <- M.[1L.., *] - f * M.[0L..0L, *]
            R.[1L.., *] <- R.[1L.., *] - f * R.[0L..0L, *]
                           
        /// Brings the matrix M into diagonal form with zero columns moved to the right end.                    
        let rec diagonalize (R: Tensor<bigint>) (C: Tensor<bigint>) (M: Tensor<bigint>) =
            if movePivot R C M then
                // non-zero columns remaining
                // Apply row and column reductions until pivot is the only non-zero element in the
                // first row and first column.
                let mutable changed = true
                while changed do
                    makePivotGCD R M false |> ignore
                    eliminateRows R M
                    changed <- makePivotGCD C.T M.T false
                    eliminateRows C.T M.T
                    
                // proceed to next row and column
                diagonalize R.[1L.., *] C.[*, 1L..] M.[1L.., 1L..]
        
        /// Ensures that M.[[i; i]] divides M.[[i+1; i+1]] for all i.
        /// Thus an element on the diagonal will divide all elements that follow it.    
        let rec makeDivChain (R: Tensor<bigint>) (C: Tensor<bigint>) (M: Tensor<bigint>) =
            if M.Shape.[0] >= 1L && M.Shape.[1] >= 1L then
                // ensure that diagonal element is positive
                if M.[[0L; 0L]] < bigint.Zero then
                    M.[*, 0L] <- bigint.MinusOne * M.[*, 0L]
                    C.[*, 0L] <- bigint.MinusOne * C.[*, 0L]
                     
            if M.Shape.[0] >= 2L && M.Shape.[1] >= 2L && M.[[0L; 0L]] <> bigint.Zero then           
                // check divisibility
                if M.[[1L; 1L]] % M.[[0L; 0L]] <> bigint.Zero then
                    // diagonal element does not divide following element
                    // add following column to this column to get non-zero entry in M.[[1L; 1L]]
                    linComb 0L 1L (bigint.One, bigint.One) (bigint.Zero, bigint.One) M.T
                    linComb 0L 1L (bigint.One, bigint.One) (bigint.Zero, bigint.One) C.T
                    // reapply diagonalization procedure 
                    diagonalize R C M
                    // M.[[0L; 0L]] is now GCD(M.[[0L; 0L]], M.[[1L; 1L]]) and the   
                    // new M.[[1L; 1L]] is a linear combination of M.[[0L; 0L]] and M.[[1L; 1L]],
                    // thus divisable by their GCD.
                    // Diagonal has to be rechecked now because it can happen that 
                    // M.[[-1L; -1L]] does not divide M.[[0; 0L]] anymore.
                    true
                else
                    // proceed with next diagonal element
                    makeDivChain R.[1L.., *] C.[*, 1L..] M.[1L.., 1L..]
            else
                // finished without change
                false

        // initialize U, S, V
        let rows, cols = getRowsCols A
        let U = Tensor.identity A.Dev rows
        let V = Tensor.identity A.Dev cols                  
        let S = Tensor.copy A        
        
        // apply Smith algorithm
        diagonalize U V S
        while makeDivChain U V S do ()
        U, S, V
        

    /// Computes the inverse I, solvability constraints S and null-space N of the specified integer matrix M,
    /// which can be of any shape and rank.
    /// The inversion is carried out over the domain of integers.
    /// The return values is a tuple (I, S, N), which fulfilles the following properties:
    /// Inverse:     M .* I .* M = M.
    /// Solvability: S .* M = 0.
    /// Null-space:  M .* N = 0.
    /// The equation system M .* x = y is solvable when S .* y = 0 and I .* y is an integer vector.
    /// In this case, the set of solutions is given by x = I .* y + N .* z where z is any integer vector.    
    let integerInverse (M: Tensor<bigint>) =
               
        // Obtain Smith Normal form, so that S = U .* M .* V.
        let U, S, V = smithNormalForm M
                
        // Compute rank, i.e. number of non-zero columns of S.
        let rec diagRank r (T: Tensor<bigint>) =
            if T.Shape.[0] = 0L || T.Shape.[1] = 0L then r
            elif T.[[0L; 0L]] = bigint.Zero then r
            else diagRank (r+1L) T.[1L.., 1L..]        
        let rank = diagRank 0L S        
        
        // We want to solve the equation system M .* x = y.
        // Using the Smith Normal Form this can be rewritten as 
        // S .* x' = y' with x = V .* x' and y' = U .* y.
        // The diagonal of S has 'rank' non-zero entries and all zero entries are at the lower right.
        
        // For a non-zero entry S_ii <> 0, we must have x'_i = y'_i / S_ii.
        let toRat = Tensor<Rat>.convert
        let nzS = S.[..rank-1L, ..rank-1L] |> Tensor.diag 
        let nzU = U.[..rank-1L, *]
        let nzV = V.[*, ..rank-1L]         
        let nzVSU = toRat nzV .* (toRat nzU / toRat nzS.[*, NewAxis])
                       
        // For each zero row S_*i = 0, we must have y'_i = 0 for the system to be solvable. 
        let zU = U.[rank.., *]   
      
        // For each zero column S_*i = 0, the corresponding x'_i can have any integer value and thus the
        // corresponding columns of V form a basis of the null-space.
        let zV = V.[*, rank..]
        
        // inverse, solvability, null-space
        nzVSU, zU, zV


    
