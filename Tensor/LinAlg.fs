namespace Tensor.Algorithms

open Tensor
open Tensor.Utils


/// Linear algebra algorithms
module LinAlg =

    let private getRowsCols (L: Tensor<_>) =
        match L.Shape with
        | [rows; cols] -> rows, cols
        | _ -> failwithf "L must be a matrix but it has shape %A" L.Shape

    /// Computes the reduced row echelon form of L augmented with matrix A.
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

    /// Computes the reduced row echelon form of L.
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
    /// then the value x = I .* y solves y = M .* x.
    /// If x contains no component from the null-space (i.e. N^T .* x = 0),
    /// then we can recover x from y = M .* x by x = I .* y.
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



        


    
