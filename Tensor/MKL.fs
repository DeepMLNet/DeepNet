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
