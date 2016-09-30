namespace Basics

open System
open System.Threading
open System.Reflection
open System.Reflection.Emit
open System.Runtime.InteropServices


module PassArrayByVal =

    let private currentDomain = Thread.GetDomain()
    let private dynAsmName = new AssemblyName()
    dynAsmName.Name <- "ArrayByValDynamicTypes"
    let private asmBuilder = currentDomain.DefineDynamicAssembly(dynAsmName, AssemblyBuilderAccess.Run)
    let private modBuilder = asmBuilder.DefineDynamicModule("Module")
    
    let mutable private arrayStructTypes : Map<string, Type> = Map.empty


    /// creates a struct type containing a fixed size array of given type and size
    let private arrayStructType (valueType: Type) (size: int) =
        let typeName = sprintf "%s_%dElems" valueType.Name size

        match Map.tryFind typeName arrayStructTypes with
        | Some typ -> typ
        | None ->
            lock modBuilder (fun () ->
                // define new value type with attribute [<Struct; StructLayout(LayoutKind.Sequential)>]
                let tb = modBuilder.DefineType(typeName, 
                                               TypeAttributes.Public ||| TypeAttributes.SequentialLayout,
                                               typeof<ValueType>);

                // define public fields "DataXXX" of type valueType
                for i = 0 to size - 1 do
                    let fn = sprintf "Data%d" i
                    tb.DefineField(fn, valueType, FieldAttributes.Public) |> ignore

                // create defined type and cache it
                let typ = tb.CreateType()
                arrayStructTypes <- arrayStructTypes |> Map.add typeName typ
                typ
            )

    /// passes an array by value to a native method
    let passArrayByValue (data: 'T []) =
        // create struct 
        let strctType = arrayStructType typeof<'T> (Array.length data)
        let strct = Activator.CreateInstance(strctType)

        // set data
        for i = 0 to Array.length data - 1 do
            strct.GetType().GetField(sprintf "Data%d" i).SetValue(strct, data.[i])

        // test if marshalling works
        //let size = Marshal.SizeOf strct
        //let mem = Marshal.AllocHGlobal size
        //Marshal.StructureToPtr (strct, mem, false)
        //Marshal.FreeHGlobal mem

        strct


        

        
