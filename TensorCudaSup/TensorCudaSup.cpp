// TensorCudaSup.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "TensorCudaSup.h"


// This is an example of an exported variable
TENSORCUDASUP_API int nTensorCudaSup=0;

// This is an example of an exported function.
TENSORCUDASUP_API int fnTensorCudaSup(void)
{
    return 42;
}

// This is the constructor of a class that has been exported.
// see TensorCudaSup.h for the class definition
CTensorCudaSup::CTensorCudaSup()
{
    return;
}
