<#
.SYNOPSIS
Build and verify the single-file Windows installer for MicroSeg Desktop.

.DESCRIPTION
Builds an onedir PyInstaller application containing the windowed desktop
launcher and console CLI companion, smoke-tests both launchers, compiles one
offline Inno Setup executable, silently installs it to a verification folder,
smoke-tests the installed application, uninstalls it, and writes SHA-256 release
metadata.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/build_windows_installer.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/build_windows_installer.ps1 -InstallBuildDependencies
#>

[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$InstallBuildDependencies,
    [switch]$SkipTests,
    [switch]$SkipInstaller,
    [switch]$SkipPackagedSmokeTest,
    [switch]$SkipInstallerVerification
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host $Label
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Invoke-CheckedGuiProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    # Windows PowerShell does not wait for GUI-subsystem executables invoked
    # with `&`. Quote each argument explicitly and use Start-Process -Wait so
    # smoke reports and installer outputs are complete before validation.
    $argumentLine = ($Arguments | ForEach-Object {
        '"' + ([string]$_).Replace('"', '\"') + '"'
    }) -join ' '
    Write-Host $Label
    $process = Start-Process `
        -FilePath $Executable `
        -ArgumentList $argumentLine `
        -PassThru `
        -Wait `
        -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($process.ExitCode)."
    }
}

function Resolve-PythonExecutable {
    param([string]$Requested)

    if ($Requested) {
        if (Test-Path -LiteralPath $Requested) {
            return (Resolve-Path -LiteralPath $Requested).Path
        }
        $command = Get-Command $Requested -ErrorAction Stop
        return $command.Source
    }

    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return (Resolve-Path -LiteralPath $venvPython).Path
    }
    return (Get-Command python -ErrorAction Stop).Source
}

function Resolve-InnoCompiler {
    $command = Get-Command iscc -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "Inno Setup 6 compiler (ISCC.exe) was not found. Install Inno Setup 6 or use -SkipInstaller."
}

$python = Resolve-PythonExecutable -Requested $PythonExe
$version = (& $python -c "from src.microseg.version import __version__; print(__version__)" | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or -not $version) {
    throw "Unable to read the MicroSeg version from src.microseg.version."
}

Write-Host "Repository root: $repoRoot"
Write-Host "Python: $python"
Write-Host "Release version: $version"

if (-not $SkipTests) {
    Invoke-CheckedCommand `
        -Label "Running release and desktop packaging tests..." `
        -Executable $python `
        -Arguments @(
            "-m", "pytest", "-q",
            "tests/test_release_v1_packaging.py",
            "tests/test_phase2_desktop_workflow.py",
            "tests/test_phase27_qt_settings_smoke.py"
        )
}

& $python -c "import PyInstaller"
if ($LASTEXITCODE -ne 0) {
    if (-not $InstallBuildDependencies) {
        throw "PyInstaller is not installed. Re-run with -InstallBuildDependencies or install requirements-build.txt."
    }
    Invoke-CheckedCommand `
        -Label "Installing pinned desktop build dependencies..." `
        -Executable $python `
        -Arguments @("-m", "pip", "install", "-r", "requirements-build.txt")
}

$distRoot = Join-Path $repoRoot "dist"
$appRoot = Join-Path $distRoot "MicroSegDesktop"
$desktopExe = Join-Path $appRoot "MicroSegDesktop.exe"
$cliExe = Join-Path $appRoot "MicroSegCLI.exe"
$installerRoot = Join-Path $distRoot "installer"
$installerPath = Join-Path $installerRoot "MicroSegDesktop_${version}_offline_setup.exe"

Invoke-CheckedCommand `
    -Label "Building desktop and CLI launchers with PyInstaller..." `
    -Executable $python `
    -Arguments @("-m", "PyInstaller", "--noconfirm", "--clean", "apps/desktop/windows/microseg_desktop.spec")

foreach ($requiredPath in @($desktopExe, $cliExe)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Expected packaged launcher is missing: $requiredPath"
    }
}

if (-not $SkipPackagedSmokeTest) {
    $smokeReport = Join-Path $distRoot "MicroSegDesktop_packaged_smoke.json"
    if (Test-Path -LiteralPath $smokeReport -PathType Leaf) {
        Remove-Item -LiteralPath $smokeReport -Force
    }
    $previousQpa = $env:QT_QPA_PLATFORM
    $env:QT_QPA_PLATFORM = "offscreen"
    try {
        Invoke-CheckedGuiProcess `
            -Label "Smoke-testing the packaged desktop application..." `
            -Executable $desktopExe `
            -Arguments @("--smoke-test", "--smoke-report", $smokeReport)
    }
    finally {
        $env:QT_QPA_PLATFORM = $previousQpa
    }
    if (-not (Test-Path -LiteralPath $smokeReport -PathType Leaf)) {
        throw "Packaged desktop smoke report was not created: $smokeReport"
    }
    $packagedSmoke = Get-Content -LiteralPath $smokeReport -Raw | ConvertFrom-Json
    if ($packagedSmoke.status -ne "passed" -or $packagedSmoke.app_version -ne $version) {
        throw "Packaged desktop smoke report did not validate release version $version."
    }
    Invoke-CheckedCommand `
        -Label "Smoke-testing the packaged CLI companion..." `
        -Executable $cliExe `
        -Arguments @("--version")
}

if ($SkipInstaller) {
    Write-Host "Installer compilation skipped. Packaged app: $appRoot"
    exit 0
}

$iscc = Resolve-InnoCompiler
New-Item -ItemType Directory -Force -Path $installerRoot | Out-Null
Invoke-CheckedCommand `
    -Label "Compiling the single-file offline installer with Inno Setup..." `
    -Executable $iscc `
    -Arguments @("/DAppVersion=$version", "apps/desktop/windows/microseg_desktop.iss")

if (-not (Test-Path -LiteralPath $installerPath -PathType Leaf)) {
    throw "Expected installer was not created: $installerPath"
}

$installedSmokeReport = ""
if (-not $SkipInstallerVerification) {
    $verificationRoot = Join-Path $installerRoot "verification"
    $verificationRootFull = [System.IO.Path]::GetFullPath($verificationRoot)
    $installerRootFull = [System.IO.Path]::GetFullPath($installerRoot)
    if (-not $verificationRootFull.StartsWith($installerRootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Installer verification path escaped the installer output root: $verificationRootFull"
    }
    if (Test-Path -LiteralPath $verificationRootFull) {
        Remove-Item -LiteralPath $verificationRootFull -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $verificationRootFull | Out-Null

    $verificationInstall = Join-Path $verificationRootFull "app"
    $installerLog = Join-Path $verificationRootFull "install.log"
    Invoke-CheckedGuiProcess `
        -Label "Silently installing the release artifact for verification..." `
        -Executable $installerPath `
        -Arguments @(
            "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOICONS",
            "/DIR=$verificationInstall", "/LOG=$installerLog"
        )

    $installedDesktop = Join-Path $verificationInstall "MicroSegDesktop.exe"
    $installedCli = Join-Path $verificationInstall "MicroSegCLI.exe"
    $installedSmokeReport = Join-Path $verificationRootFull "installed_smoke.json"
    foreach ($requiredPath in @($installedDesktop, $installedCli)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Installer verification did not produce: $requiredPath"
        }
    }

    $previousQpa = $env:QT_QPA_PLATFORM
    $env:QT_QPA_PLATFORM = "offscreen"
    try {
        Invoke-CheckedGuiProcess `
            -Label "Smoke-testing the installed desktop application..." `
            -Executable $installedDesktop `
            -Arguments @("--smoke-test", "--smoke-report", $installedSmokeReport)
    }
    finally {
        $env:QT_QPA_PLATFORM = $previousQpa
    }
    if (-not (Test-Path -LiteralPath $installedSmokeReport -PathType Leaf)) {
        throw "Installed desktop smoke report was not created: $installedSmokeReport"
    }
    $installedSmoke = Get-Content -LiteralPath $installedSmokeReport -Raw | ConvertFrom-Json
    if ($installedSmoke.status -ne "passed" -or $installedSmoke.app_version -ne $version) {
        throw "Installed desktop smoke report did not validate release version $version."
    }
    Invoke-CheckedCommand `
        -Label "Smoke-testing the installed CLI companion..." `
        -Executable $installedCli `
        -Arguments @("--version")

    $uninstaller = Join-Path $verificationInstall "unins000.exe"
    if (-not (Test-Path -LiteralPath $uninstaller -PathType Leaf)) {
        throw "Installer verification uninstaller is missing: $uninstaller"
    }
    Invoke-CheckedGuiProcess `
        -Label "Uninstalling the verification copy..." `
        -Executable $uninstaller `
        -Arguments @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART")
}

$artifact = Get-Item -LiteralPath $installerPath
$sha256 = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
$gitCommit = (& git rev-parse HEAD 2>$null | Select-Object -Last 1)
$pyInstallerVersion = (& $python -c "import PyInstaller; print(PyInstaller.__version__)" | Select-Object -Last 1).Trim()
$releaseManifest = [ordered]@{
    schema_version = "microseg.desktop_installer_release.v1"
    status = "passed"
    app_version = $version
    artifact = $artifact.Name
    bytes = [int64]$artifact.Length
    sha256 = $sha256
    built_at_utc = [DateTime]::UtcNow.ToString("o")
    source_commit = "$gitCommit".Trim()
    python_version = (& $python -c "import platform; print(platform.python_version())" | Select-Object -Last 1).Trim()
    pyinstaller_version = $pyInstallerVersion
    packaged_smoke_test = (-not $SkipPackagedSmokeTest)
    installer_verification = (-not $SkipInstallerVerification)
    installed_smoke_report = if ($installedSmokeReport) { [System.IO.Path]::GetFileName($installedSmokeReport) } else { "" }
}
$manifestPath = Join-Path $installerRoot "MicroSegDesktop_${version}_release.json"
$releaseManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Release installer verified successfully."
Write-Host "Installer: $installerPath"
Write-Host "SHA-256: $sha256"
Write-Host "Release metadata: $manifestPath"
