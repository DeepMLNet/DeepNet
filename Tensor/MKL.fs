namespace ArrayNDNS

open System
open System.Runtime.InteropServices

module MKL =

    type lapack_int = int64

    let LAPACK_ROW_MAJOR = 101
    let LAPACK_COL_MAJOR = 102


    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgetrf (int matrix_layout, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<Out>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgetrf (int matrix_layout, lapack_int m, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<Out>] lapack_int[] ipiv)


    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgetri (int matrix_layout, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<In>] lapack_int[] ipiv)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgetri (int matrix_layout, lapack_int n, 
                                      nativeint a, lapack_int lda, 
                                      [<In>] lapack_int[] ipiv)


    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_sgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n,
                                     nativeint a, lapack_int lda,
                                     nativeint wr, nativeint wi,
                                     nativeint vl, lapack_int ldvl,
                                     nativeint vr, lapack_int ldvr)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n,
                                     nativeint a, lapack_int lda,
                                     nativeint wr, nativeint wi,
                                     nativeint vl, lapack_int ldvl,
                                     nativeint vr, lapack_int ldvr)


    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_ssyevd (int matrix_layout, char jobz, char uplo, lapack_int n,
                                      nativeint a, lapack_int lda,
                                      nativeint w)

    [<DllImport("tensor_mkl.dll", CallingConvention=CallingConvention.Cdecl)>]
    extern lapack_int LAPACKE_dsyevd (int matrix_layout, char jobz, char uplo, lapack_int n,
                                      nativeint a, lapack_int lda,
                                      nativeint w)


