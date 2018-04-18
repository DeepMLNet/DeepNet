#!/usr/bin/pwsh
# Script to build DocFX documentation.

Push-Location $PSScriptRoot

if (!(Test-Path -Path docfx)) {
    ./Acquire.ps1
}

./docfx/docfx.console/docfx.exe metadata

Push-Location generate
dotnet run
Pop-Location

./docfx/docfx.console/docfx.exe

Pop-Location

