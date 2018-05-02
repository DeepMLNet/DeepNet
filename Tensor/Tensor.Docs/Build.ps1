#!/usr/bin/pwsh
# Script to build DocFX documentation.

param(
    [switch] $noGenerate = $false
)

$BenchmarkVersion = "0.4.9"

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot

if (!(Test-Path -Path docfx)) {
    ./Acquire.ps1
}

if (!$noGenerate) {
    if ($PSVersionTable.Platform -eq "Unix") {
        mono ./docfx/docfx.console/tools/docfx.exe metadata
    } else {
        ./docfx/docfx.console/tools/docfx.exe metadata
    }    
    Push-Location generate
    dotnet run
    Pop-Location

    dotnet run -p ../Tensor.Benchmark/Report -- --output benchmarks/benchmark-linux.html ../Tensor.Benchmark/Results/Linux-Tensor-$BenchmarkVersion ../Tensor.Benchmark/Results/Linux-NumPy-Anaconda3-5.1.0
    dotnet run -p ../Tensor.Benchmark/Report -- --output benchmarks/benchmark-windows.html ../Tensor.Benchmark/Results/Windows-Tensor-$BenchmarkVersion ../Tensor.Benchmark/Results/Windows-NumPy-Anaconda3-5.1.0
}

if ($PSVersionTable.Platform -eq "Unix") {
    mono ./docfx/docfx.console/tools/docfx.exe
} else {
    ./docfx/docfx.console/tools/docfx.exe
}    

Pop-Location

