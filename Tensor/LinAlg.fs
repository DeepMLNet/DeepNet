namespace Tensor.Algorithms

open Tensor
open Tensor.Utils


/// Tools for computing and working with the row echelon form of a matrix.
module RowEchelonForm =

    /// Computes the reduced row echelon form of L augmented with matrix A.
    let computeAugmented (L: Tensor<'T>) (A: Tensor<'T>) =
        let rows, cols =
            match L.Shape with
            | [rows; cols] -> rows, cols
            | _ -> failwithf "L must be a matrix but it has shape %A" L.Shape
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
                //printfn "-------- GE step with active row=%d col=%d:\nR=\n%A\nB=\n%A" row col R B

                // find pivot row by maximum magnitude
                let pivot = 
                    R.[row.., col] 
                    |> abs 
                    |> Tensor.argMax 
                    |> List.exactlyOne
                    |> fun p -> p + row
                //printfn "Using row %d as pivot." pivot

                // swap active row with pivot row
                swapRows R row pivot
                swapRows B row pivot
                //printfn "After swap:\nR=\n%A\nB=\n%A" R B

                let pivotVal = R.[row, col] |> Tensor.copy
                if Tensor.value pivotVal <> zero<'T> then
                    // make active row start with a one
                    R.[row, *] <- R.[row, *] / pivotVal
                    B.[row, *] <- B.[row, *] / pivotVal   
                    //printfn "After division:\nR=\n%A\nB=\n%A" R B

                    // eliminate active column from all other rows
                    let facs = R.[0L .. row-1L, col..col] |> Tensor.copy
                    R.[0L .. row-1L, *] <- R.[0L .. row-1L, *] - facs * R.[row..row, *]
                    B.[0L .. row-1L, *] <- B.[0L .. row-1L, *] - facs * B.[row..row, *]
                    let facs = R.[row+1L .., col..col] |> Tensor.copy
                    R.[row+1L .., *] <- R.[row+1L .., *] - facs * R.[row..row, *]
                    B.[row+1L .., *] <- B.[row+1L .., *] - facs * B.[row..row, *]
                    //printfn "After elimination:\nR=\n%A\nB=\n%A" R B

                    // continue with next row and column
                    step unnormalizedCols (row+1L) (col+1L)
                else
                    // All remaining entries in active column are zero,
                    // thus it cannot be normalized.
                    //printfn "Pivot is zero."
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
    let compute (L: Tensor<'T>) =
        let A = Tensor.zeros L.Dev [L.Shape.[0]; 0L]
        let R, nonZeroRows, unnormalizedCols, _ = computeAugmented L A
        R, nonZeroRows, unnormalizedCols


    let pseudoInvert (L: Tensor<'T>) =
        let rows, cols =
            match L.Shape with
            | [rows; cols] -> rows, cols
            | _ -> failwithf "L must be a matrix but it has shape %A" L.Shape
        
        // compute row echelon form
        let A = HostTensor.identity (max rows cols) |> fun A -> A.[0L .. rows-1L, *]       
        let E, nzRows, unCols, I = computeAugmented L A

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
        let LI = E.T .* I

        // null space of L
        let N = -E.T .* U
        for c, r in List.indexed unCols do
            N.[[r; int64 c]] <- Tensor.One

        // solvability constraint
        let S = I.[nzRows.., *]

        LI, S, N



        


    
