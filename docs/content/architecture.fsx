(**

Architecture
============

The following diagram shows a bird's eye view of Deep.Net's architecture.

![Architecture diagram](img/Architecture.png)


Shapes and sizes
================

Deep.Net uses symbolic sizes and shapes to verify the correctness of expressions while they are being defined.
For this purpose it contains types that allow basic algebraic operations on symbols.
Code for that functionality is located in `SymTensor/SymShape.fs`.

Size specification
------------------

The size of each dimension of a symbolic tensor is represented by the type `SizeSpecT`.
A size specification can either be a numeric size (for example `3`), a symbolic size variable (for example `n`) or a polynomial in symbolic size variables (for example `3 * n**2 + 2 * m`).
Size specifications support and are closed under the operations `+`, `-`, `*` and `Pow` (with an integer exponent).
They can be compared using the operators `%=` (equal) or `.=` (equal ignoring broadcastability).
Supporting functions are located in the module `SizeSpec`.

The supporting polynomial algebra is implemented by `SizeProductT` and `SizeMultinomT`.


Shape specification
-------------------

A shape specification `ShapeSpecT` is a list of size specifications `SizeSpecT`.
Some supporting functions are located in the module `ShapeSpec`.


Range specification
-------------------

Symbolic ranges (for selecting elements from and slicing tensors) are provided by the type `RangeSpecT`.
Tensors can be sliced using symbolic sizes `SizeSpecT` or dynamic values, i.e. the range can depend on the result of another expression.
At the moment, however, the length of the range must be a symbolic size known at model compile time.


### Simple range specification
During expression compilation, range specifications are transformed into simple range specification `SimpleRangeSpecT`.



Symbolic expressions
====================

TODO

Expression validation
---------------------


Automatic differentiation
=========================

TODO


Compilation pipeline
====================

TODO






*)