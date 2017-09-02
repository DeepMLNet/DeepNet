namespace Tensor.Algorithms

open Tensor
open Tensor.Utils


/// Tools for computing and working with the row echelon form of a matrix.
module RowEchelonForm =

    /// Computes the reduced row echelon form of L.
    let compute (L: Tensor<'T>) (augmentations: Tensor<'T> list) =
        let rows, cols =
            match L.Shape with
            | [rows; cols] -> rows, cols
            | _ -> failwithf "L must be a matrix but it has shape %A" L.Shape
        for A in augmentations do
            match A.Shape with
            | [ar; _] when ar = rows -> ()
            | _ -> 
                failwith "all augmentations must be matrices with same number of rows as L"
        let R = Tensor.copy L

        // step of Gaussian Elimination algroithm
        let rec step row col =
            if row < rows && col < cols then
                //printfn "-------- GE step with active row=%d col=%d:\n%A" row col R

                // find pivot row by maximum magnitude
                let pivot = 
                    R.[row.., col] 
                    |> abs 
                    |> Tensor.argMax 
                    |> List.exactlyOne
                    |> fun p -> p + row
                //printfn "Using row %d as pivot." pivot

                // swap active row with pivot row
                let tmp = R.[row, *] |> Tensor.copy
                R.[row, *] <- R.[pivot, *]
                R.[pivot, *] <- tmp
                //printfn "After swap:\n%A" R

                let pivotVal = R.[row, col]
                if Tensor.value pivotVal <> zero<'T> then
                    // make active row start with a one
                    R.[row, *] <- R.[row, *] / pivotVal
                    //printfn "After division:\n%A" R

                    // eliminate active column from all other rows
                    R.[0L .. row-1L, *] <- 
                        R.[0L .. row-1L, *] - R.[0L .. row-1L, col..col] * R.[row..row, *]
                    R.[row+1L .., *] <- 
                        R.[row+1L .., *] - R.[row+1L .., col..col] * R.[row..row, *]
                    //printfn "After elimination:\n%A" R

                    // continue with next row and column
                    step (row+1L) (col+1L)
                else
                    // all entries in active column are zero
                    // try next column
                    printfn "Pivot is zero."
                    step row (col+1L)

        step 0L 0L
        R


    
