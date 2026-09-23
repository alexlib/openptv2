# Launch the OpenPTV2 GUI in a browser via Xpra, from native PowerShell on
# Windows (Docker Desktop, WSL2 or Hyper-V backend — either works, this
# talks to the Docker Desktop daemon either way). Equivalent of
# run-xpra.sh; use that one instead from WSL2/Git Bash/macOS/Linux.
#
#   .\docker\run-xpra.ps1                              # mount $PWD, port 9876
#   .\docker\run-xpra.ps1 -DataDir C:\Users\me\ptv      # mount a specific folder
#   $env:OPENPTV2_XPRA_PASSWORD = "secret"; .\docker\run-xpra.ps1
#
# Then open http://localhost:9876 in any browser.
#
# Windows-specific notes (see docs/browser-gui.md for the full version):
#   - Docker Desktop must have the drive containing -DataDir shared
#     (Settings -> Resources -> File Sharing), or the mount will appear
#     empty inside the container.
#   - Pass a normal Windows path (C:\Users\...); Docker Desktop translates
#     it, no manual conversion to /c/... or similar needed here.
#   - No OPENPTV2_XPRA_PASSWORD set? Binds to 127.0.0.1 only and runs with
#     no xpra auth — fine for local use, do not change the port bind below
#     to 0.0.0.0 without also setting a password.

param(
    [string]$DataDir = (Get-Location).Path,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $DataDir -PathType Container)) {
    Write-Error "Not a directory: $DataDir"
    exit 1
}
$DataDir = (Resolve-Path $DataDir).Path

$Image = if ($env:OPENPTV2_IMAGE) { $env:OPENPTV2_IMAGE } else { "openptv2-xpra" }

$EnvArgs = @("-e", "QT_X11_NO_MITSHM=1")
if ($env:OPENPTV2_XPRA_PASSWORD) {
    $EnvArgs += @("-e", "OPENPTV2_XPRA_PASSWORD=$($env:OPENPTV2_XPRA_PASSWORD)")
} else {
    $EnvArgs += @("-e", "OPENPTV2_XPRA_ALLOW_INSECURE=1")
    Write-Warning "No OPENPTV2_XPRA_PASSWORD set - running with no auth, bound to 127.0.0.1 only. Set `$env:OPENPTV2_XPRA_PASSWORD to add a login prompt."
}

$TtyArgs = if ([Console]::IsInputRedirected) { @("-i") } else { @("-it") }

$DockerArgs = @("run", "--rm") + $TtyArgs + @(
    "-p", "127.0.0.1:9876:9876"
) + $EnvArgs + @(
    "-v", "${DataDir}:/data",
    $Image
)
if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $DockerArgs += $ExtraArgs
}

docker @DockerArgs
