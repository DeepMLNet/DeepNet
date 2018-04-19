namespace Tensor.Host

open System
open System.Reflection
open System.Numerics
open System.Threading.Tasks
open System.Linq.Expressions
open System.Collections.Generic
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

open Tensor
open Tensor.Utils
open Tensor.Backend



/// BLAS / LAPACK library imports
module internal BLAS =

    type lapack_int = int64
    type MKL_INT = int64

    type CBLAS_LAYOUT =
        | CblasRowMajor = 101
        | CblasColMajor = 102
    type CBLAS_TRANSPOSE =
        | CblasNoTrans = 111
        | CblasTrans = 112
        | CblasConjTrans = 113
    type CBLAS_UPLO = 
        | CblasUpper = 121
        | CblasLower = 122
    
    let LAPACK_ROW_MAJOR = 101
    let LAPACK_COL_MAJOR = 102

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern single cblas_sdot (MKL_INT n, nativeint x, MKL_INT incx, nativeint y, MKL_INT incy)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern double cblas_ddot (MKL_INT n, nativeint x, MKL_INT incx, nativeint y, MKL_INT incy)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_sgemv (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                             MKL_INT M, MKL_INT N, single alpha,
                             nativeint A, MKL_INT lda, nativeint X, MKL_INT incx,
                             single beta, nativeint Y, MKL_INT incy)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_dgemv (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                             MKL_INT M, MKL_INT N, double alpha,
                             nativeint A, MKL_INT lda, nativeint X, MKL_INT incx,
                             double beta, nativeint Y, MKL_INT incy)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_sgemm (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                             MKL_INT M, MKL_INT N, MKL_INT K, single alpha,
                             nativeint A, MKL_INT lda, nativeint B, MKL_INT ldb,
                             single beta, nativeint C, MKL_INT ldc)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_dgemm (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                             MKL_INT M, MKL_INT N, MKL_INT K, double alpha,
                             nativeint A, MKL_INT lda, nativeint B, MKL_INT ldb,
                             double beta, nativeint C, MKL_INT ldc)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_sgemm_batch (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE[] TransA, CBLAS_TRANSPOSE[] TransB,
                                   MKL_INT[] M, MKL_INT[] N, MKL_INT[] K, single[] alpha,
                                   nativeint[] A, MKL_INT[] lda, nativeint[] B, MKL_INT[] ldb,
                                   single[] beta, nativeint[] C, MKL_INT[] ldc,
                                   MKL_INT group_count, MKL_INT[] group_size)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern void cblas_dgemm_batch (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE[] TransA, CBLAS_TRANSPOSE[] TransB,
                                   MKL_INT[] M, MKL_INT[] N, MKL_INT[] K, double[] alpha,
                                   nativeint[] A, MKL_INT[] lda, nativeint[] B, MKL_INT[] ldb,
                                   double[] beta, nativeint[] C, MKL_INT[] ldc,
                                   MKL_INT group_count, MKL_INT[] group_size)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgetrf (int matrix_layout, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<Out>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgetrf (int matrix_layout, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<Out>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgetri (int matrix_layout, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<In>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgetri (int matrix_layout, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<In>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n,
                                     nativeint a, lapack_int lda,
                                     nativeint wr, nativeint wi,
                                     nativeint vl, lapack_int ldvl,
                                     nativeint vr, lapack_int ldvr)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n,
                                     nativeint a, lapack_int lda,
                                     nativeint wr, nativeint wi,
                                     nativeint vl, lapack_int ldvl,
                                     nativeint vr, lapack_int ldvr)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_ssyevd (int matrix_layout, char jobz, char uplo, lapack_int n,
                                      nativeint a, lapack_int lda,
                                      nativeint w)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dsyevd (int matrix_layout, char jobz, char uplo, lapack_int n,
                                      nativeint a, lapack_int lda,
                                      nativeint w)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgesdd (int matrix_layout, char jobz, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      nativeint s, 
                                      nativeint u, lapack_int ldu, 
                                      nativeint vt, lapack_int ldvt)

    [<DllImport("tensor_mkl", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgesdd (int matrix_layout, char jobz, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      nativeint s, 
                                      nativeint u, lapack_int ldu, 
                                      nativeint vt, lapack_int ldvt)


[<AutoOpen>]
module internal HostBLASExtensions = 
    type Tensor.Backend.BLAS.MatrixInfo with
        member this.CTrans = 
            match this.Trans with
            | BLAS.NoTrans   -> BLAS.CBLAS_TRANSPOSE.CblasNoTrans
            | BLAS.Trans     -> BLAS.CBLAS_TRANSPOSE.CblasTrans
            | BLAS.ConjTrans -> BLAS.CBLAS_TRANSPOSE.CblasConjTrans
