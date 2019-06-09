namespace Tensor.Expr.Base

open System
open System.Reflection
open DeepNet.Utils
open Tensor.Expr


/// Declares that the type is extending an op by implementing additional interfaces.
/// The type's constructor is called with the op instance as argument.
[<AttributeUsage(AttributeTargets.Class)>]
type OpExtenderAttribute () =
    inherit System.Attribute()


/// Methods for obtaining op extender instances.
module OpExtender =

    type private OpInterface = OpInterface of op:TypeName * iface:TypeName with
        static member from<'I> (op: IOp) =
            let op = TypeName.ofObject op
            let iface = TypeName.ofType<'I>
            OpInterface (op, iface)

    let mutable private extenders: Map<OpInterface, Type> = Map.empty

    let private registerExt opIface extType =
        match extenders |> Map.tryFind opIface with
        | Some prevType ->
            failwithf "Cannot register %A as extender for %A because it has already been registered to %A."
                      extType opIface prevType
        | None ->
            extenders <- extenders |> Map.add opIface extType                 

    /// Gets the op extender of the specified op for interface 'I, if it exists.
    let tryGet<'I> (op: IOp) =
        match box op with
        | :? 'I as ext -> Some ext
        | _ ->
            let opIface = OpInterface.from<'I> op
            match extenders.TryFind opIface with
            | Some extType ->
                let ext = Activator.CreateInstance (extType, [|box op|]) :?> 'I
                Some ext
            | None ->
                None

    /// Registers all op extenders from the specified assembly.
    let register (asm: Assembly) =
        let extTypes =
            asm.GetTypes()
            |> Seq.filter (fun typ ->
                typ.CustomAttributes 
                |> Seq.exists (fun a -> a.AttributeType = typeof<OpExtenderAttribute>))

        for extType in extTypes do
            let opType = 
                extType.GetConstructors() 
                |> Seq.tryPick (fun cons ->
                    let args = cons.GetParameters()
                    if args.Length = 1 then
                        let arg = args.[0]
                        Some arg.ParameterType
                    else
                        None)
            match opType with
            | Some opType ->
                for iface in extType.GetInterfaces() do
                    registerExt (OpInterface (TypeName.ofTypeInst opType, TypeName.ofTypeInst iface)) extType
            | None -> ()

    do register (Assembly.GetExecutingAssembly())

