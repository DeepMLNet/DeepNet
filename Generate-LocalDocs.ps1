 if (-not ((Test-Path packages\FSharp.Formatting.2.14.2) -and (Test-Path packages\FAKE.4.27.0))) {
    echo "Installing..."
    .\dist\nuget.exe install FSharp.Formatting -outputdirectory packages -Version 2.14.2 -Verbosity quiet
    .\dist\nuget.exe install FAKE -outputdirectory packages -Version 4.27.0 -Verbosity quiet
}

& "C:\Program Files (x86)\Microsoft SDKs\F#\4.0\Framework\v4.0\Fsi.exe"  --define:HELP docs\tools\generate.fsx
