namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open Tensor.Utils
open Tensor



type HDF5Tests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let ``Open HDF5 file for writing`` () =
        use f = HDF5.OpenWrite "test.h5"
        ()

    [<Fact>]
    let ``Write and read tensor`` () =
        let tensor = HostTensor.counting 123L
        (
            use f = HDF5.OpenWrite "write_test.h5"
            HostTensor.write f "tensor" tensor
        )
        (
            use f = HDF5.OpenRead "write_test.h5"
            let tread = HostTensor.read f "tensor"
            Tensor.almostEqual (tensor, tread) |> should equal true
        )

    [<Fact>]
    let ``Get group entries`` () =
        let tensor = HostTensor.counting 123L
        (
            use f = HDF5.OpenWrite "write_test.h5"
            HostTensor.write f "/tensor" tensor
            HostTensor.write f "/group/a" tensor
            HostTensor.write f "/group/b" tensor
        )
        (
            use f = HDF5.OpenRead "write_test.h5"
            let entries = f.Entries "/"
            printfn "Entries at /: %A" entries
            Set entries |> should equal (Set [HDF5Entry.Dataset "tensor"; HDF5Entry.Group "group"])
            let groupEntries = f.Entries "/group"
            printfn "Entries at /group: %A" groupEntries
            Set groupEntries |> should equal (Set [HDF5Entry.Dataset "a"; HDF5Entry.Dataset "b"])
        )    

    [<Fact>]
    let ``Attributes`` () =
        let tensor = HostTensor.counting 123L
        (
            use f = HDF5.OpenWrite "write_test.h5"
            HostTensor.write f "/tensor" tensor
            f.SetAttribute ("/tensor", "str", "strValue")
            f.SetAttribute ("/tensor", "int", 123)
            f.SetAttribute ("/tensor", "float", 123.0)
        )

        (
            use f = HDF5.OpenRead "write_test.h5"

            let strValue: string = f.GetAttribute ("/tensor", "str") 
            strValue |> should equal "strValue"
            let intValue: int = f.GetAttribute ("/tensor", "int") 
            intValue |> should equal 123
            let floatValue: float = f.GetAttribute ("/tensor", "float") 
            floatValue |> should equal 123.0

            let atrs = f.Attributes "/tensor"
            printfn "Attributes of /tensor: %A" atrs
            let atrsExpected = Map [
                "str", typeof<string>
                "int", typeof<int>
                "float", typeof<float>
            ]
            atrs |> should equal atrsExpected
        )