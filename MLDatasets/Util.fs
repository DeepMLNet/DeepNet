namespace Datasets

open ArrayNDNS
open SymTensor

module Util =

    let nonScalar (a: ArrayNDT<'T>) =
        let shp = ArrayND.shape a
        if (shp.Length <= 0) then
            ArrayND.reshape [(ArrayND.nElems a)] a
        else
            a

