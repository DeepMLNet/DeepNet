module CudaCheck

// need something that is as simple as eval for CUDA
// problem: it makes sense for an expression to return multiple results
// so a simple return might not be so nice

// possible interfaces:
//  1. the thenao way: compile a function with specified inputs and outputs
// would it fit the f# way? how to nicely define such a function with static types?
// each 


// let exprValsFunc = exprToFunc (expr1, expr2) par1 par2 par3
// let exprVal1, exprVal = exprValsFunc parVal1 parVal2 parVal3

// how to integrate with variable setters, i.e. training?
// can we prespecify variables, i.e. explicit currying?

// add support for that in expression?
// hmm, not mathematically sensible: an expression does not contain values
// perhaps it is best to go without a global state at all


// 2. the explicit map way
//   each function expects an explicit map of variable names to values, i.e. EvalEnv

// perhaps a combination of both? i.e. args and kwargs?
// but this is not easily possible in f#


// could perhaps also use type providers?
// don't see the advantage right now

// we could add support for global variables in expressions (possibly with namespaces)
// would need to add Scope option to VarSpecT


// so plan:
// method 1 is just a wrapper around method 2 that builds an EvalEnv from the specified parameters
//
// we also need to know the sizes before we can compile...
// so when to compile?
//   - on first invocation with explicit parameters, so that sizes can be determined automatically?
//   - sizes can vary, so we need multiple instantions of the CudaRecipe for each size
//  when to dispose?
//   - should a compiled function be disposable?
//     - makes sense for resource control but complicates programming
//     - so probably not
//     - would call Dispose of instantiaed CudeExecs in Finalizer


// TODO:
// - implement method 2
// - reuse simple wrapper
// generic evaluation frontend with different backends
// i.e. specify exprs and get functions
// problem: CUDA evaluator needs additional information, i.e. variable location (host/device)
//          but this is not really a problem because it can be derived from the variables when they are passed in






let compareEvalWithCudaEval evalEnv expr =
    ()  