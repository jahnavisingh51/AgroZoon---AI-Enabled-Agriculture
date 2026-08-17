Param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir "venv\Scripts\python.exe"
$appPath = Join-Path $scriptDir "app\app.py"
$reqPath = Join-Path $scriptDir "requirements.txt"

function Ensure-Venv {
    if (Test-Path $venvPython) { return }
    Write-Host "Creating virtual environment in .\venv ..."
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        & py -m venv (Join-Path $scriptDir "venv")
    } else {
        & python -m venv (Join-Path $scriptDir "venv")
    }
    if (-not (Test-Path $venvPython)) {
        throw "Failed to create venv at $venvPython"
    }
}

if (-not (Test-Path $appPath)) {
    Write-Host "Application file not found at $appPath"
    exit 1
}

if (-not (Test-Path $reqPath)) {
    Write-Host "requirements.txt not found at $reqPath"
    exit 1
}

Ensure-Venv

Write-Host "Installing/updating dependencies..."
& $venvPython -m pip install --upgrade pip | Out-Host
& $venvPython -m pip install -r $reqPath | Out-Host

Write-Host "Starting AgroZoon Flask app..."
& $venvPython $appPath

