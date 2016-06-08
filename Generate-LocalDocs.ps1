if (-not ((Test-Path packages\FSharp.Formatting) -and (Test-Path packages\FAKE))) {
    echo "Installing..."
    .\dist\nuget.exe install FSharp.Formatting -outputdirectory packages -Verbosity quiet -ExcludeVersion
    .\dist\nuget.exe install FAKE -outputdirectory packages -Verbosity quiet -ExcludeVersion
}

& "C:\Program Files (x86)\Microsoft SDKs\F#\4.0\Framework\v4.0\Fsi.exe"  --define:HELP docs\tools\generate.fsx
