namespace LangRNN

open System
open System.Text.RegularExpressions

open MargieBot
open MargieBot.Responders
open MargieBot.Models

open Basics
open ArrayNDNS


type SlackBot (data:      WordData,
               model:     GRUInst,
               slackKey:  string) =

    let maxPars = 3
    let maxLines = 15

    let bot = Bot()

    let (|Int|_|) str =
       match System.Int32.TryParse(str) with
       | (true,int) -> Some(int)
       | _ -> None

    let preprocess (line: string) = 
        let line = line.ToLower()
        let line = line.Replace("\"", "").Replace("-", "").Replace("!", ".").Replace(",", "").Replace(".", "")
        let line = line.Replace("„", "").Replace("“", "").Replace("…", "").Replace("–", "").Replace("=", "")
        let line = line.Replace(":", "").Replace("refrain", "").Replace("—", "")
        let line = line.Replace("ref","").Replace("courus","").Replace("corus", "").Replace("chorus", "")
        let line = line.Replace(">", "").Replace("<", "").Replace(" x ", "").Replace("&", " ").Replace("#", "")
        let line = Regex.Replace(line, @"(\w)'(\w)", "$1 '$2")
        let line = Regex.Replace(line, @"(\w)'", "$1")
        let line = Regex.Replace(line, @"\[.*\]", "")
        let line = Regex.Replace(line, @"\(.*\)", "")
        let line = Regex.Replace(line, @"\d", "")
        let line = Regex.Replace(line, @" [^ ia] ", " ")
        let line = Regex.Replace(line, @"(\w)\?", "$1 ?")
        let line = Regex.Replace(line, @"^ ", "")
        let line = Regex.Replace(line, @"\n", " > ")
        let line = Regex.Replace(line, @"  ", " ")
        let line = line.Trim()
        ">" :: (line.Split(' ') |> List.ofArray)

    do bot.Responders.Add (
        {
            new IResponder with
                member this.CanRespond context =
                    context.Message.MentionsBot ||
                    context.Message.ChatHub.Type = SlackChatHubType.DM

                member this.GetResponse context =
                    // set worker thread's CUDA context
                    Cuda.CudaSup.setContext ()

                    let msg = context.Message.Text
                    let msg = Regex.Replace(msg, @"<@.+>", "").Trim()
                    //printfn "Got message: %s" msg_

                    // extract seed if specified
                    let words = msg.Split(' ') |> List.ofArray
                    let seed, words =
                        match words with
                        | (Int seed)::rWords -> seed, rWords
                        | _ -> 0, words

                    // preprocess and split into words
                    let line = words |> String.concat " "
                    let words = preprocess line

                    try
                        // tokenize
                        let startTokens = 
                            words 
                            |> data.Tokenize 
                            |> ArrayNDHost.ofList
                            |> ArrayND.reshape [1L; -1L]
                            |> ArrayNDCuda.toDev

                        // generate and detokenize
                        let genTokens = model.Generate seed {Words=startTokens}
                        let genTokens = genTokens.Words.[0L, *] |> ArrayNDHost.fetch 
                        let genWords = genTokens |> List.ofSeq |> data.Detokenize 

                        // format response
                        let mutable pars = 0
                        let mutable lines = 0 
                        let response = 
                            words @ genWords
                            |> List.takeWhile (function 
                                               | ">" -> lines <- lines + 1; lines < maxLines
                                               | "---" -> pars <- pars + 1; pars < maxPars
                                               | _ -> true)
                            |> List.map (function
                                         | ">" -> "\n>"
                                         | "---" -> "\n "
                                         | w -> w)
                            |> String.concat " "

                        new BotMessage(Text=sprintf "%s\n`%d`" response seed)
                    with UnknownWord uw ->
                        new BotMessage (Text=sprintf "I don't understand the word %s." uw)
        })

    do
        bot.Connect(slackKey).Wait()
        assert bot.IsConnected


