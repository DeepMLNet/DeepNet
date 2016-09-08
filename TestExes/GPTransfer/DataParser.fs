namespace GPTransfer

open Basics
open ArrayNDNS
open SymTensor
open Datasets
open System.IO
open System.Text.RegularExpressions

module DataParser =
    
    ///Loads a dataSet and saves the lines in a string array
    let stringsToDataStr inStr =
        let type1 = Regex.Match(inStr, "((\w+.\w+)|(\w+)),")
        if type1.Success then
            let elems = type1.Groups
            let nElems = elems.Count
            let outArray = Array.create nElems ""
            elems.CopyTo(outArray, 0)
            outArray
        else
            let type2 = Regex.Match(inStr, "((\w+.\w+)|(\w+)|(\w+.)|(\".+\"))\s+")
            if type2.Success then
                let elems = type2.Groups
                let nElems = elems.Count
                let outArray = Array.create nElems ""
                elems.CopyTo(outArray, 0)
                outArray
            else
                failwith "unknown data representation"
    
    ///Generates a 2D array from an array of arrays
    let MultiArray (inp:'a[][]) = 
        let count = inp |> Array.length
        //if arrays have different lengths they are cut at the length of the shortest one
        let rows =  [0..count-1] 
                    |> List.map (fun x -> inp.[x] |> Array.length)
                    |> List.min
        Array2D.init count rows (fun i j -> inp.[j].[i])

    ///Takes a 2D string array where each row is a sample
    ///Returns an array of floats and a list of option dictionarrys
    ///which contain mappings from non numerical strings to class numbers
    let strToNumArys strAry =
        let colNum = Array2D.length1 strAry
        let rowNum = Array2D.length2 strAry
        let numAry = Array2D.create colNum rowNum 0.0
        let dictionarys = Array.create colNum None
        for c= 0 to colNum-1 do
            let m = Regex.Match( strAry.[c,0], "((\d+.\d+)|(\d+)|(\d+.))")
            if m.Success then
                for r = 0 to rowNum - 1 do
                    numAry.[c,r] <- float strAry.[c,r]
            else
                let dict = (Dictionary<string,int>())
                let mutable count = 0
                for r = 0 to rowNum - 1 do
                    let s = strAry.[c,r]
                    let suc,numval = dict.TryGetValue s
                    if suc then
                        numAry.[c,r] <-float numval
                    else
                        numAry.[c,r] <-float count
                        dict.Add(s, count)
                        count <- count + 1
                dictionarys.[c] <- Some dict
        numAry, dictionarys

    
    let fileToNumArray path = 
        let txtArys = File.ReadAllLines path
        let strArys = txtArys |> Array.map stringsToDataStr |> MultiArray
        strArys |> strToNumArys
   
    let fileToData path (tgt: list<int>) =
        let numAry, dictAry = fileToNumArray path
        let colNum = Array2D.length1 numAry
        let sampleNum = Array2D.length2 numAry
        let maxIdx = List.max tgt
        let sizeT = List.length tgt
        let sizeX = colNum - sizeT
        
        if colNum <= maxIdx then
            failwithf  "dataset contains %d <= %d columns" colNum maxIdx
        if List.min tgt < 0 then
            failwithf  "%d < 0 and therefore no valid index" (List.min tgt)
        
        let tArray = Array2D.create sizeT sampleNum 0.0
        let xArray = Array2D.create (colNum - sizeT) sampleNum 0.0
        let tDicts = Array.create sizeT None
        let xDicts = Array.create (colNum - sizeT) None
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
                    tArryTemp.[idxT] <- numAry.[c,s]
                else
                    let _,idxX = xDict.TryGetValue c
                    xArryTemp.[idxX] <- numAry.[c,s]

            outList <- List.append outList [xArryTemp |> ArrayNDHost.ofArray,tArryTemp |> ArrayNDHost.ofArray ]
        outList , (xDicts,tDicts)
    
    type floatSample ={
        Input:  ArrayNDHostT<float>;
        Target: ArrayNDHostT<float>
        }

    type singleSample ={
        InputS:  ArrayNDHostT<single>;
        TargetS: ArrayNDHostT<single>
        }
    
    ///Loads a file and returns a dataset of floats and two lists of dictionarrys that match non numeric values to class indices
    ///all columns of the file with indices in tgt are target values, the rest are input values
    let loadFloatDataset path (tgt: list<int>)=
        let data, dicts = fileToData path tgt
        let data = data |> List.map (fun (x,y) -> {Input = x;Target= y})
        let dataSet = data |> Dataset.FromSamples
        dataSet,dicts

    ///Loads a file and returns a dataset of singles and two lists of dictionarrys that match non numeric values to class indices
    ///all columns of the file with indices in tgt are target values, the rest are input values
    let loadSingleDataset path (tgt: list<int>)=
        let data, dicts = fileToData path tgt
        let data = data |> List.map (fun (x,y) -> 
            {InputS = x |> ArrayNDHost.toArray |> Array.map (fun x -> conv<single> x) |> ArrayNDHost.ofArray;
            TargetS = y |> ArrayNDHost.toArray |> Array.map (fun x -> conv<single> x) |> ArrayNDHost.ofArray})
        let dataSet = data |> Dataset.FromSamples
        dataSet,dicts