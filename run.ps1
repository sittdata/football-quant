<#
.SYNOPSIS
    Runs the fbpredict pipeline using a Python interpreter this machine has.

.DESCRIPTION
    The bare `python` command on this machine resolves to the Windows Store
    stub, so this wrapper locates a real Python 3 and calls it.

.EXAMPLE
    .\run.ps1 demo
    .\run.ps1 show-config
    .\run.ps1 backtest --threshold 0.05
    .\run.ps1 test
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $Command,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)

if ($null -eq $Rest) { $Rest = @() }

function Get-PythonExe {
    $candidates = @()

    if ($env:LOCALAPPDATA) {
        $root = Join-Path $env:LOCALAPPDATA 'Programs\Python'
        if (Test-Path -LiteralPath $root) {
            $candidates += Get-ChildItem -LiteralPath $root -Filter 'python.exe' -Recurse -ErrorAction SilentlyContinue |
                Sort-Object -Property FullName -Descending |
                Select-Object -ExpandProperty FullName
        }
    }

    $candidates += 'python'

    foreach ($candidate in $candidates) {
        try {
            $major = & $candidate -c 'import sys; print(sys.version_info[0])' 2>$null
        }
        catch {
            continue
        }
        if ("$major" -eq '3') { return $candidate }
    }

    throw 'No Python 3 interpreter found. Install Python 3 or edit run.ps1.'
}

$python = Get-PythonExe
Write-Host "python: $python" -ForegroundColor DarkGray

if (-not $Command) {
    Write-Host 'usage: .\run.ps1 <command> [args]' -ForegroundColor Yellow
    Write-Host '  commands: demo, backtest, test, show-config, fetch-results, probe-odds, link, ...' -ForegroundColor Yellow
    & $python -m fbpredict --help
    exit 0
}

if ($Command -eq 'test') {
    & $python -m pytest @Rest
}
else {
    & $python -m fbpredict $Command @Rest
}

exit $LASTEXITCODE
