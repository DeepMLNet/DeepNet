#!/usr/bin/pwsh
# Script to build DocFX documentation.

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot

if (!(Test-Path -Path docfx)) {
    ./Acquire.ps1
}

if ($PSVersionTable.Platform -eq "Unix") {
    $executor = "mono"
} else {
    $executor = ""
}

& $executor ./docfx/docfx.console/docfx.exe metadata

Push-Location generate
dotnet run
Pop-Location

& $executor ./docfx/docfx.console/docfx.exe

Pop-Location

