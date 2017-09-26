namespace Tensor.Algorithms

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
            if List.isEmpty unCols then Tensor.zeros<'T> L.Dev [rows; 0L]
            else
                unCols
                |> List.map (fun c -> E.[*, c..c])
                |> Tensor.concat 1

        // zero unnormalized columns in row echelon form E
        // (setting undetermined values of solution to zero)
        for c in unCols do
            E.[*, c] <- Tensor.scalar E.Dev Tensor.Zero
        
        // calculate inverse of image
        let LI = (E.T .* I).[*, 0L .. rows-1L]

        // null space of L
        let N = -E.T .* U
        for c, r in List.indexed unCols do
            N.[[r; int64 c]] <- Tensor.One

        // solvability constraint
        let S = I.[nzRows.., 0L .. rows-1L]

        LI, S, N


    /// Computes the Smith normal form of integer matrix A.
    let smithNormalForm (A: Tensor<bigint>) =       
        let rows, cols = getRowsCols A                  
        let A = Tensor.copy A
             
        /// Moves a non-zero column to the left and swaps rows so that the 
        /// element in the upper-left corner of the returned matrix is non-zero.
        /// M is modified in-place. Returns None if M is zero.   
        let movePivot (M: Tensor<bigint>) =
            // find left-most column with at least a non-zero entry
            match M <<>> bigint.Zero |> Tensor.anyAxis 0 |> Tensor.tryFind true with
            | Some [nzCol] ->
                // move non-zero column to the left
                if nzCol <> 0L then
                    M.[*, 0L] <- M.[*, nzCol]
                    M.[*, nzCol].FillConst bigint.Zero
                
                // find first row that has non-zero element in left-most column
                let nzRow = M.[*, 0L] <<>> bigint.Zero |> Tensor.find true |> List.exactlyOne
                // and swap that row with the first row, if necessary
                if nzRow <> 0L then
                    let tmp = Tensor.copy M.[0L, *] 
                    M.[0L, *] <- M.[nzRow, *]
                    M.[nzRow, *] <- tmp
                    
                true
            | None -> 
                // all columns are zero
                false
            | _ -> failwith "unexpected find result"                              

        /// Ensures that the pivot element in the upper-left corner of M divides the 
        /// first element of all following rows.
        /// This is done by replacing rows by linear combinations of other rows.
        let rec makePivotGCD (M: Tensor<bigint>) changed =
            let pivot = M.[[0L; 0L]]           
            // find any row that is not divisable by pivot
            let R = M.[*, 0L] % pivot
            match R <<>> bigint.Zero |> Tensor.tryFind true with
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
                let M0, MO = Tensor.copy M.[0L, *], Tensor.copy M.[ndRow, *]
                M.[0L, *] <- sigma * M0 + tau * MO
                M.[ndRow, *] <-  -gamma * M0 + alpha * MO
            
                // iterate until all rows are divisible by pivot
                makePivotGCD M true
            | None ->
                // all rows are divisable by pivot
                changed
            | _ -> failwith "unexpected find result"
        
        /// Ensures that the first element of all but the first row is zero by substracting
        /// an appropriate multiple of the first row from these rows.
        let eliminateRows (M: Tensor<bigint>) =
            let f = M.[1L.., 0L..0L] / M.[0L, 0L]
            M.[1L.., *] <- M.[1L.., *] - f * M.[0L..0L, *]
                           
        /// Brings the matrix M into diagonal form with zero columns moved to the right end.                    
        let rec diagonalize (M: Tensor<bigint>) =
            if movePivot M then
                // non-zero columns remaining
                // Apply row and column reductions until pivot is the only non-zero element in the
                // first row and first column.
                let mutable changed = true
                while changed do
                    makePivotGCD M false |> ignore
                    eliminateRows M
                    changed <- makePivotGCD M.T false
                    eliminateRows M.T
                    
                // proceed to next row and column
                diagonalize M.[1L.., 1L..]
        
        /// Ensures that M.[[i; i]] divides M.[[i+1; i+1]] for all i.
        /// Thus an element on the diagonal will divide all elements that follw it.    
        let rec makeDivChain (startM: Tensor<bigint>) (M: Tensor<bigint>) =
            if M.Shape.[0] >= 1L && M.Shape.[1] >= 1L then
                // ensure that diagonal element is positive
                if M.[[0L; 0L]] < bigint.Zero then
                    M.[*, 0L] <- bigint.MinusOne * M.[*, 0L]
                     
            if M.Shape.[0] >= 2L && M.Shape.[1] >= 2L && M.[[0L; 0L]] <> bigint.Zero then           
                // check divisibility
                if M.[[1L; 1L]] % M.[[0L; 0L]] <> bigint.Zero then
                    // diagonal element does not divide following element
                    // add following column to this column to get non-zero entry in M.[[1L; 1L]]
                    M.[*, 0L] <- M.[*, 0L] + M.[*, 1L]
                    // reapply diagonalization procedure 
                    diagonalize M
                    // M.[[0L; 0L]] is now GCD(M.[[0L; 0L]], M.[[1L; 1L]]) and the   
                    // new M.[[1L; 1L]] is a linear combination of M.[[0L; 0L]] and M.[[1L; 1L]],
                    // thus divisable by their GCD.
                    // Diagonal has to be rechecked now because it can happen that 
                    // M.[[-1L; -1L]] does not divide M.[[0; 0L]] anymore.
                    makeDivChain startM startM
                else
                    // proceed with next diagonal element
                    makeDivChain startM M.[1L.., 1L..]

        diagonalize A
        makeDivChain A A
        A
        


    
