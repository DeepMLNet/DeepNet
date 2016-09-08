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
                let dict = (Dictionary<string,int>())
                let mutable count = 0
                for r = 0 to rowNum - 1 do
                    let s = strAry.[r,c]
                    //check if string is already in dictionary
                    let suc,numval = dict.TryGetValue s
                    if suc then
                        //replace string with class index
                        numAry.[r,c] <-float numval
                    else
                        //add to dictionary with new class index
                        numAry.[r,c] <-float count
                        dict.Add(s, count)
                        count <- count + 1
                dictionarys.[c] <- Some dict
        numAry, dictionarys

    ///Reads a file and returns a 2D-array of floats
    let fileToNumArray path separator= 
        let txtArys = File.ReadAllLines path
        let strArys = txtArys |> Array.map (stringsToDataStr separator) |> MultiArray
        strArys |> strToNumArys
   
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
    let loadFloatDataset path (tgt: list<int>) separator=
        let data, dicts = fileToData path tgt separator
        let data = data |> List.map (fun (x,y) -> {Input = x;Target= y})
        let dataSet = data |> Dataset.FromSamples
        dataSet,dicts

    ///Loads a file and returns a dataset of singles and two lists of dictionarrys that match non numeric values to class indices
    ///all columns of the file with indices in tgt are target values, the rest are input values
    let loadSingleDataset path (tgt: list<int>) separator=
        let data, dicts = fileToData path tgt separator
        let data = data |> List.map (fun (x,y) -> 
            {InputS = x |> ArrayNDHost.toArray |> Array.map (fun x -> conv<single> x) |> ArrayNDHost.ofArray;
            TargetS = y |> ArrayNDHost.toArray |> Array.map (fun x -> conv<single> x) |> ArrayNDHost.ofArray})
        let dataSet = data |> Dataset.FromSamples
        dataSet,dicts