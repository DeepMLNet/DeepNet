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
            // define new value type with attribute [<Struct; StructLayout(LayoutKind.Sequential)>]
            let tb = modBuilder.DefineType(typeName, 
                                           TypeAttributes.Public ||| TypeAttributes.SequentialLayout,
                                           typeof<ValueType>);

            // define public field "Data" of type valueType[]
            let arrayType = valueType.MakeArrayType()       
            let fb = tb.DefineField("Data", arrayType, FieldAttributes.Public)

            // set MarshalAs(UnmanagedType.ByValArray, SizeConst=size) attribute on field
            let cTorInfo = typeof<MarshalAsAttribute>.GetConstructor([| typeof<UnmanagedType> |])
            let sizeConstInfo = typeof<MarshalAsAttribute>.GetField("SizeConst")
            let ab = CustomAttributeBuilder(cTorInfo, [| UnmanagedType.ByValArray :> obj |],
                                            [| sizeConstInfo |], [| box size |])                
            fb.SetCustomAttribute(ab)

            // create defined type and cache it
            let typ = tb.CreateType()
            arrayStructTypes <- arrayStructTypes |> Map.add typeName typ
            typ


    /// passes an array by value to a native method
    let passArrayByValue (data: 'T []) =
        // create struct 
        let strctType = arrayStructType typeof<'T> (Array.length data)
        let strct = Activator.CreateInstance(strctType)

        // set data
        strct.GetType().GetField("Data").SetValue(strct, data)

        strct


        

        
