namespace SymTensor

module Function =
    ()


//type CudaEvalFunction (expr) =

    // function instance depends on sizes and locations of variables
    // necessary information is SizeSymbolEnv and CudaEnvT
    // can we have a generic VarEnvT with variables stored in different places?
    // hmmm, difficult, because CPU backend does not know about GPU
    // possibilites: allow VarEnvT to contain CPU and GPU variables
    // then CPU backend needs to know about GPU variables which introduces unnecessary complexity
    // make VarEnv somehow generic so that it can be instantiated to contain CPU only
    // and CPU/GPU variables
    // other possibility: make NDArray directly GPU aware (using a single type)
    // which is kind of nice
    // would allow funny things like memory mapping to GPU (but can also be done with two classes)
    // can at least 



    //let getInstanceForEvalEnv evalEnv =
        


//let buildFun builder expr =
  //  builder expr

