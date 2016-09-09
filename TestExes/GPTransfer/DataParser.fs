namespace GPTransfer

open Basics
open ArrayNDNS
open SymTensor
open Datasets
open System
open System.IO
open System.Text.RegularExpressions

module DataParser =
    
    ///Loads a dataSet and saves the lines in a string array
    let stringsToDataStr separator (inStr:string)=
        let split = inStr.Split [|separator|]
        split
        
    ///Generates a 2D array from an array of arrays
    let MultiArray (inp:'a[][]) = 
        let count = inp |> Array.length
        //if arrays have different lengths they are cut at the length of the shortest one
        let rows =  inp
                    |> Seq.map (fun x -> x |> Array.length)
                    |> Seq.min
        Array2D.init count rows (fun i j -> inp.[i].[j])

    ///Takes a 2D string array where each row is a sample
    ///Returns an array of floats and a list of option dictionarrys
    ///which contain mappings from non numerical strings to class numbers
    let strToNumArys strAry =
        let rowNum = Array2D.length1 strAry
        let colNum = Array2D.length2 strAry
        let numAry = Array2D.create rowNum colNum 0.0
        let dictionarys = Array.create colNum None
        //iterate through all columns
        for c= 0 to colNum-1 do
            //check if the elements are number
            //TODO: can not yet handle symbols for missing data
            let m = Regex.Match( strAry.[0,c], "((\d+.\d+)|(\d+)|(\d+.))")
            if m.Success then
                //cast all elements of column to float
                for r = 0 to rowNum - 1 do
                    numAry.[r,c] <- float strAry.[r,c]
            else
                //replace all strings with a class index
                //create a dictionary that maps the strings to int values
                let dict = (Dictionary<string,float>())
                let mutable count = 0.0
                for r = 0 to rowNum - 1 do
                    let s = strAry.[r,c]
                    //check if string is already in dictionary
                    let suc,numval = dict.TryGetValue s
                    if suc then
                        //replace string with class index
                        numAry.[r,c] <- numval
                    else
                        //add to dictionary with new class index
                        numAry.[r,c] <- count
                        dict.Add(s, count)
                        count <- count + 1.0
                dictionarys.[c] <- Some dict
        numAry, dictionarys

    ///Reads a file and returns a 2D-array of floats
    let fileToNumArray path separator= 
        let txtArys = File.ReadAllLines path
        let strArys = txtArys |> Array.map (stringsToDataStr separator) |> MultiArray
        strArys |> strToNumArys
    
    let deleteArrayIdx idx (ary:'a[]) =
        let length = Array.length ary
        if length - 1 > idx then
            if idx > 0 then
                Array.append ary.[..(idx - 1)] ary.[(idx + 1)..]
            else
                ary.[1..]
        else if length - 1 = idx then
            if idx = 0 then
                [||]
            else
                ary.[..(idx - 1)]
        else
            failwithf "array index %d is out of bounds" idx

    let oneHotEnc   (smpLst: list<float[][]>)
                    idx 
                    (dicts:Collections.Generic.Dictionary<string,float> option[]) =
        let nSamples = List.length smpLst
        let dLength = Array.length dicts
        let sLength = smpLst.[0].[idx] |> Array.length
        if sLength <> dLength then
            failwithf "size of dictionary %d <> number of samples %d" dLength sLength
        let mutable offset = 0
        for d = 0 to dLength - 1 do
            let dictO = dicts.[d]
            match dictO with
            | None -> ()
            | Some dict ->
                let nClasses = dict.Count
                let sLength = smpLst.[0].[idx] |> Array.length
                for s = 0 to nSamples - 1 do
                    let oldArray = smpLst.[s].[idx]
                    let newArray = Array.create nClasses 0.0
                    let newArray = Array.append oldArray newArray
                    let classIdx = int oldArray.[d - offset]
                    newArray.[sLength - 1 + classIdx] <- 1.0
                    smpLst.[s].[idx] <- newArray |> deleteArrayIdx d
                offset <- offset + 1
        smpLst

    let dataOneHotEncoding data dict =
        let (xDict,tDict) = dict
        let data = oneHotEnc data 0 xDict
        let data = oneHotEnc data 1 tDict
        data

    ///Reads a file and returns data as nd Arrays
    ///In:      path: locaton of the file
    ///         tgt: indices of columns that should be contained in the target array
    ///         seperator: character that seperates the columns in the file
    ///Out:     outList (ArrayNDHostT<float>*ArrayNDHostT<float>)
    ///             samples containing input and target values
    ///         (xDict,yDict) Array of Dictionary option for each column in input and training data
    let fileToData path (tgt: list<int>) separator =
        let numAry, dictAry = fileToNumArray path separator
        let sampleNum = Array2D.length1 numAry
        let colNum = Array2D.length2 numAry
        let maxIdx = List.max tgt
        let sizeT = List.length tgt
        let sizeX = colNum - sizeT
        
        if colNum <= maxIdx then
            failwithf  "dataset contains %d <= %d columns" colNum maxIdx
        if List.min tgt < 0 then
            failwithf  "%d < 0 and therefore no valid index" (List.min tgt)
        
        let tArray = Array2D.create sampleNum sizeT 0.0
        let xArray = Array2D.create sampleNum sizeX 0.0
        let tDicts = Array.create sizeT None
        let xDicts = Array.create sizeX None
        let mutable xcount = 0
        let xDict =  (Dictionary<int,int>())
        let tDict =  (Dictionary<int,int>())
        
        for c=0 to colNum - 1 do
            if List.contains c tgt then
                let idx = List.findIndex (fun x -> x = c) tgt
                tDict.Add(c,idx)
                tDicts.[idx] <- dictAry.[c]
            else
                xDict.Add(c,xcount)
                xDicts.[xcount] <- dictAry.[c]
                xcount <- xcount + 1  
        let mutable outList = []

        for s= 0 to sampleNum - 1 do
            let xArryTemp = Array.create sizeX 0.0
            let tArryTemp = Array.create sizeT 0.0
            for c = 0 to colNum - 1 do
                let sucT,idxT = tDict.TryGetValue c          
                if sucT then
                    tArryTemp.[idxT] <- numAry.[s,c]
                else
                    let _,idxX = xDict.TryGetValue c
                    xArryTemp.[idxX] <- numAry.[s,c]

            outList <- List.append outList [[|xArryTemp ;tArryTemp|]]
        dataOneHotEncoding outList  (xDicts,tDicts)
    

    type floatSample ={
        Input:  ArrayNDT<float>;
        Target: ArrayNDT<float>
        } with
            /// copies this dataset to the CUDA GPU
            member this.toCuda () =
                {Input = this.Input :?> ArrayNDHostT<float> |> ArrayNDCuda.toDev;
                Target = this.Target :?> ArrayNDHostT<float> |> ArrayNDCuda.toDev}

    type singleSample ={
        InputS:  ArrayNDT<single>;
        TargetS: ArrayNDT<single>
        } with
            /// copies this dataset to the CUDA GPU
            member this.ToCuda () =
                {InputS = this.InputS :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev
                 TargetS = this.TargetS :?> ArrayNDHostT<single> |> ArrayNDCuda.toDev}
    
    ///Loads a file and returns a dataset of floats and two lists of dictionarrys that match non numeric values to class indices
    ///all columns of the file with indices in tgt are target values, the rest are input values
    let loadFloatDataset path (tgt: list<int>) separator=
        let data = fileToData path tgt separator
        let data = data |> List.map (fun x ->  {Input = x.[0] |> ArrayNDHost.ofArray;
                                                Target= x.[1] |> ArrayNDHost.ofArray})
        let dataSet = data |> Dataset.FromSamples
        dataSet

    ///Loads a file and returns a dataset of singles and two lists of dictionarrys that match non numeric values to class indices
    ///all columns of the file with indices in tgt are target values, the rest are input values
    let loadSingleDataset path (tgt: list<int>) separator=
        let data = fileToData path tgt separator
        let data = data |> List.map (fun x -> 
            {InputS = x.[0]  |> Array.map (fun x -> conv<single> x) |> ArrayNDHost.ofArray;
            TargetS = x.[1]  |> Array.map (fun x -> conv<single> x) |> ArrayNDHost.ofArray})
        let dataSet = data |> Dataset.FromSamples
        dataSet