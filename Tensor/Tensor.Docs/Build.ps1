#!/usr/bin/pwsh
# Script to build DocFX documentation.

param(
    [switch] $noGenerate = $false
)

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

if (!$noGenerate) {
    & $executor ./docfx/docfx.console/tools/docfx.exe metadata
    Push-Location generate
    dotnet run
    Pop-Location
}

& $executor ./docfx/docfx.console/tools/docfx.exe

Pop-Location

