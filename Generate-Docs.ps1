# $ErrorActionPreference = "Stop"

.\dist\nuget.exe install FSharp.Formatting -outputdirectory packages -Verbosity quiet -ExcludeVersion
.\dist\nuget.exe install FAKE -outputdirectory packages -Verbosity quiet -ExcludeVersion

if (Test-Path docs\output) {
    Rm -Recurse docs\output    
}

& "C:\Program Files (x86)\Microsoft SDKs\F#\4.0\Framework\v4.0\Fsi.exe"  --define:HELP --define:REFERENCE --define:RELEASE docs\tools\generate.fsx
