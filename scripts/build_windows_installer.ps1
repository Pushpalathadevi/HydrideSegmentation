<#
.SYNOPSIS
Build the Windows offline installer for MicroSeg Desktop.

.DESCRIPTION
1) Installs PyInstaller if needed.
2) Builds the Qt desktop executable using apps/desktop/windows/microseg_desktop.spec.
3) Optionally compiles a single offline installer .exe with Inno Setup (iscc).

Usage:
  powershell -ExecutionPolicy Bypass -File scripts/build_windows_installer.ps1
  powershell -ExecutionPolicy Bypass -File scripts/build_windows_installer.ps1 -SkipInstaller
#>

param(
    [string]$PythonExe = "python",
    [switch]$SkipInstaller,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Repository root: $repoRoot"

if (-not $SkipSmokeTest) {
    Write-Host "Running packaging smoke test (desktop workflow test)..."
    & $PythonExe -m pytest tests/test_phase2_desktop_workflow.py
}

Write-Host "Ensuring PyInstaller is installed..."
& $PythonExe -m pip install --upgrade pyinstaller

Write-Host "Building executable with PyInstaller..."
& $PythonExe -m PyInstaller --noconfirm --clean apps/desktop/windows/microseg_desktop.spec

if ($SkipInstaller) {
    Write-Host "Skipping installer compile. Built executable is under dist/MicroSegDesktop."
    exit 0
}

$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $iscc) {
    Write-Warning "Inno Setup compiler (iscc) not found. Install Inno Setup and rerun without -SkipInstaller."
    exit 0
}

Write-Host "Compiling offline installer with Inno Setup..."
& $iscc.Source "apps/desktop/windows/microseg_desktop.iss"

Write-Host "Installer output:"
Write-Host "  dist/installer/MicroSegDesktop_0.22.0_offline_setup.exe"

