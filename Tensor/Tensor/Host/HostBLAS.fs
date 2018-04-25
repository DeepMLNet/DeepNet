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



/// BLAS / LAPACK library 
module BLAS =

    type lapack_int = int64
    type blas_int = int64

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

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __cblas_sdot = delegate of n:blas_int * x:nativeint * incx:blas_int * y:nativeint * incy:blas_int -> single

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __cblas_ddot = delegate of n:blas_int * x:nativeint * incx:blas_int * y:nativeint * incy:blas_int -> double

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __cblas_sgemv = delegate of layout:CBLAS_LAYOUT * transA:CBLAS_TRANSPOSE *
                                     m:blas_int * n:blas_int * alpha:single *
                                     a:nativeint * lda:blas_int * x:nativeint * incx:blas_int *
                                     beta:single * y:nativeint * incy:blas_int -> unit

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __cblas_dgemv = delegate of layout:CBLAS_LAYOUT * transA:CBLAS_TRANSPOSE *
                                     m:blas_int * n:blas_int * alpha:double *
                                     a:nativeint * lda:blas_int * x:nativeint * incx:blas_int *
                                     beta:double * y:nativeint * incy:blas_int -> unit

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __cblas_sgemm = delegate of layout:CBLAS_LAYOUT * transA:CBLAS_TRANSPOSE * transB:CBLAS_TRANSPOSE *
                                     m:blas_int * n:blas_int * k:blas_int * alpha:single * 
                                     a:nativeint * lda:blas_int * b:nativeint * ldb:blas_int *
                                     beta:single * c:nativeint * ldc:blas_int -> unit

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __cblas_dgemm = delegate of layout:CBLAS_LAYOUT * transA:CBLAS_TRANSPOSE * transB:CBLAS_TRANSPOSE *
                                     m:blas_int * n:blas_int * k:blas_int * alpha:double * 
                                     a:nativeint * lda:blas_int * b:nativeint * ldb:blas_int *
                                     beta:double * c:nativeint * ldc:blas_int -> unit

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __cblas_sgemm_batch = delegate of layout:CBLAS_LAYOUT * transA:CBLAS_TRANSPOSE[] * transB:CBLAS_TRANSPOSE[] *
                                           m:blas_int[] * n:blas_int[] * k:blas_int[] * alpha:single[] *
                                           a:nativeint[] * lda:blas_int[] * b:nativeint[] * ldb:blas_int[] *
                                           beta:single[] * c:nativeint[] * ldc:blas_int[] *
                                           group_count:blas_int * group_size:blas_int[] -> unit

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __cblas_dgemm_batch = delegate of layout:CBLAS_LAYOUT * transA:CBLAS_TRANSPOSE[] * transB:CBLAS_TRANSPOSE[] *
                                           m:blas_int[] * n:blas_int[] * k:blas_int[] * alpha:double[] *
                                           a:nativeint[] * lda:blas_int[] * b:nativeint[] * ldb:blas_int[] *
                                           beta:double[] * c:nativeint[] * ldc:blas_int[] *
                                           group_count:blas_int * group_size:blas_int[] -> unit

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __LAPACKE_getrf = delegate of matrix_layout:int * m:lapack_int * n:lapack_int *
                                       a:nativeint * lda:lapack_int * 
                                       [<Out>] ipiv:lapack_int[] -> lapack_int 

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __LAPACKE_getri = delegate of matrix_layout:int * n:lapack_int * 
                                       a:nativeint * lda:lapack_int * 
                                       [<Out>] ipiv:lapack_int[] -> lapack_int                                         

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __LAPACKE_geev = delegate of matrix_layout:int * jobvl:char * jobvr:char * n:lapack_int *
                                      a:nativeint * lda:lapack_int *
                                      wr:nativeint * wi:nativeint *
                                      vl:nativeint * ldvl:lapack_int *
                                      vr:nativeint * ldvr:lapack_int -> lapack_int 

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __LAPACKE_syevd = delegate of matrix_layout:int * jobz:char * uplo:char * n:lapack_int *
                                       a:nativeint * lda:lapack_int *
                                       w:nativeint -> lapack_int

    [<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
    type __LAPACKE_gesdd = delegate of matrix_layout:int * jobz:char * m:lapack_int * n:lapack_int *
                                       a:nativeint * lda:lapack_int *
                                       s:nativeint *
                                       u:nativeint * ldu:lapack_int *
                                       vt:nativeint * ldvt:lapack_int -> lapack_int


    /// BLAS/LAPACK native library 
    type Impl (blasName: NativeLibName, lapackName: NativeLibName) =

        let blas = new NativeLib (blasName)
        let lapack = new NativeLib (lapackName)

        let _cblas_sdot = blas.Func<__cblas_sdot> "cblas_sdot"
        let _cblas_ddot = blas.Func<__cblas_ddot> "cblas_ddot"
        let _cblas_sgemv = blas.Func<__cblas_sgemv> "cblas_sgemv"
        let _cblas_dgemv = blas.Func<__cblas_dgemv> "cblas_dgemv"
        let _cblas_sgemm = blas.Func<__cblas_sgemm> "cblas_sgemm"
        let _cblas_dgemm = blas.Func<__cblas_dgemm> "cblas_dgemm"
        let _cblas_sgemm_batch = blas.LazyFunc<__cblas_sgemm_batch> "cblas_sgemm_batch" 
        let _cblas_dgemm_batch = blas.LazyFunc<__cblas_dgemm_batch> "cblas_dgemm_batch"

        let _LAPACKE_sgetrf = lapack.Func<__LAPACKE_getrf> "LAPACKE_sgetrf"
        let _LAPACKE_dgetrf = lapack.Func<__LAPACKE_getrf> "LAPACKE_dgetrf"
        let _LAPACKE_sgetri = lapack.Func<__LAPACKE_getri> "LAPACKE_sgetri"
        let _LAPACKE_dgetri = lapack.Func<__LAPACKE_getri> "LAPACKE_dgetri"
        let _LAPACKE_sgeev = lapack.Func<__LAPACKE_geev> "LAPACKE_sgeev"
        let _LAPACKE_dgeev = lapack.Func<__LAPACKE_geev> "LAPACKE_dgeev"
        let _LAPACKE_ssyevd = lapack.Func<__LAPACKE_syevd> "LAPACKE_ssyevd"
        let _LAPACKE_dsyevd = lapack.Func<__LAPACKE_syevd> "LAPACKE_dsyevd"
        let _LAPACKE_sgesdd = lapack.Func<__LAPACKE_gesdd> "LAPACKE_sgesdd"
        let _LAPACKE_dgesdd = lapack.Func<__LAPACKE_gesdd> "LAPACKE_dgesdd"

        member __.cblas_sdot args = _cblas_sdot.Invoke args
        member __.cblas_ddot args = _cblas_ddot.Invoke args
        member __.cblas_sgemv args = _cblas_sgemv.Invoke args
        member __.cblas_dgemv args = _cblas_dgemv.Invoke args
        member __.cblas_sgemm args = _cblas_sgemm.Invoke args
        member __.cblas_dgemm args = _cblas_dgemm.Invoke args
        member __.cblas_sgemm_batch args = _cblas_sgemm_batch.Invoke args
        member __.cblas_dgemm_batch args = _cblas_dgemm_batch.Invoke args
        member val Has_cblas_gemm_batch = blas.HasFunc "cblas_sgemm_batch" && blas.HasFunc "cblas_dgemm_batch"

        member __.LAPACKE_sgetrf args = _LAPACKE_sgetrf.Invoke args
        member __.LAPACKE_dgetrf args = _LAPACKE_dgetrf.Invoke args
        member __.LAPACKE_sgetri args = _LAPACKE_sgetri.Invoke args
        member __.LAPACKE_dgetri args = _LAPACKE_dgetri.Invoke args
        member __.LAPACKE_sgeev args = _LAPACKE_sgeev.Invoke args
        member __.LAPACKE_dgeev args = _LAPACKE_dgeev.Invoke args
        member __.LAPACKE_ssyevd args = _LAPACKE_ssyevd.Invoke args
        member __.LAPACKE_dsyevd args = _LAPACKE_dsyevd.Invoke args
        member __.LAPACKE_sgesdd args = _LAPACKE_sgesdd.Invoke args
        member __.LAPACKE_dgesdd args = _LAPACKE_dgesdd.Invoke args



/// BLAS / LAPACK library 
type BLAS private () =

    static let load () =
        match Cfg.BLASLib with
        | BLASLib.Vendor -> BLAS.Impl (NativeLibName.Translated "blas", NativeLibName.Translated "lapacke")
        | BLASLib.IntelMKL -> BLAS.Impl (NativeLibName.Packaged "tensor_mkl", NativeLibName.Packaged "tensor_mkl")
        | BLASLib.OpenBLAS -> BLAS.Impl (NativeLibName.Translated "openblas", NativeLibName.Translated "openblas")
        | BLASLib.Custom (blas, lapack) -> BLAS.Impl (blas, lapack)

    static let mutable impl = load ()

    static do Cfg.BLASLibChangedEvent.Add (fun _ -> impl <- load())

    /// Access to actual implementation 
    static member F = impl


[<AutoOpen>]
module internal HostBLASExtensions = 
    type Tensor.Backend.BLAS.MatrixInfo with
        member this.CTrans = 
            match this.Trans with
            | BLAS.NoTrans   -> BLAS.CBLAS_TRANSPOSE.CblasNoTrans
            | BLAS.Trans     -> BLAS.CBLAS_TRANSPOSE.CblasTrans
            | BLAS.ConjTrans -> BLAS.CBLAS_TRANSPOSE.CblasConjTrans
