# Start Hearth: the monitoring server, the dashboard and the Sim Studio, all on one port.
#
#   .\scripts\start.ps1                 # http://127.0.0.1:8000/  (dashboard; Sim Studio at /studio/)
#   .\scripts\start.ps1 -Lan            # ...also reachable from the Jetson on this network
#   .\scripts\start.ps1 -Camera 0       # ...plus this laptop's webcam as a live camera source
#   .\scripts\start.ps1 -Replay         # ...plus recorded clips replayed as replay-01 (no Jetson needed)
#   .\scripts\start.ps1 -Replay -People # ...the same clips as two people at once (one lies down)
#   .\scripts\start.ps1 -Dev            # ...plus a Vite dev server with hot reload on :5173
#
# It installs the dashboard's packages and rebuilds it when its sources changed (-NoBuild skips
# that). Face recognition turns on when .env.face-demo exists (scripts/face_id_demo.sh makes it).
# Everything runs in this window; Ctrl+C stops all of it.
param(
    [int]$Port = 8000,
    [switch]$Lan,
    [int]$Camera = -1,
    [switch]$Replay,
    [switch]$People,
    [switch]$Dev,
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$frontend = Join-Path $root 'frontend'
$studio = Join-Path $frontend 'room-view'
$url = "http://127.0.0.1:$Port"

function Fail([string]$message) {
    Write-Host ''
    Write-Host $message -ForegroundColor Red
    exit 1
}
function Quote([string]$value) { if ($value -match '\s') { "`"$value`"" } else { $value } }

# Newest modification time among a UI's sources, to rebuild only when they changed.
function Newest([string[]]$paths) {
    $times = foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path -PathType Leaf) { (Get-Item -LiteralPath $path).LastWriteTimeUtc }
        elseif (Test-Path -LiteralPath $path) {
            Get-ChildItem -LiteralPath $path -Recurse -File | Where-Object { $_.FullName -notmatch '\\(node_modules|dist|generated)\\' } |
                ForEach-Object { $_.LastWriteTimeUtc }
        }
    }
    $times | Sort-Object | Select-Object -Last 1
}
function Prepare-Ui([string]$name, [string]$dir, [string]$page, [string[]]$sources) {
    # npm and Vite write progress to stderr; judge them by exit code, not by stderr.
    $ErrorActionPreference = 'Continue'
    $built = Join-Path $dir "dist\$page"
    if ($NoBuild) {
        if (-not (Test-Path -LiteralPath $built)) { Fail "The $name has not been built yet. Run without -NoBuild once." }
        return
    }
    if (-not (Test-Path -LiteralPath (Join-Path $dir 'node_modules'))) {
        Write-Host "Installing the $name's packages..."
        Push-Location $dir
        try { & npm.cmd ci --no-audit --no-fund; if ($LASTEXITCODE) { & npm.cmd install --no-audit --no-fund } }
        finally { Pop-Location }
        if ($LASTEXITCODE) { Fail "Could not install the $name's packages (npm exit $LASTEXITCODE)." }
    }
    if ((Test-Path -LiteralPath $built) -and (Get-Item -LiteralPath $built).LastWriteTimeUtc -ge (Newest $sources)) { return }
    Write-Host "Building the $name..."
    Push-Location $dir
    try { & npm.cmd run build } finally { Pop-Location }
    if ($LASTEXITCODE) { Fail "The $name failed to build (npm exit $LASTEXITCODE). See the errors above." }
}

# --- prerequisites ---------------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $python)) {
    Fail ("No Python environment. From the repository root, once:`n" +
          "  python -m venv .venv`n" +
          "  .\.venv\Scripts\python.exe -m pip install -e ./shared/python -e './backend[dev,face-demo]'`n" +
          "For the webcam or replay with the posture model also install './backend[vision]' (see README).")
}
$ErrorActionPreference = 'Continue'
& $python -c 'import fastapi, uvicorn, hearth_schema' 2>$null
$ErrorActionPreference = 'Stop'
if ($LASTEXITCODE) { Fail "The .venv is missing server packages. Run: .\.venv\Scripts\python.exe -m pip install -e ./shared/python -e './backend[dev,face-demo]'" }
if (-not $NoBuild -or $Dev) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { Fail 'Node.js (npm) is needed to build the dashboard: https://nodejs.org/ . Or pass -NoBuild to use an existing build.' }
}
try {
    $probe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
    $probe.Start(); $probe.Stop()
} catch {
    $running = $null
    try { $running = Invoke-RestMethod "$url/health" -TimeoutSec 2 } catch {}
    if ($running) { Fail "Hearth is already running on port $Port ($url/). Stop it first, or use -Port." }
    Fail "Port $Port is in use by another program. Free it or choose another with -Port."
}

# --- dashboard and Sim Studio ----------------------------------------------------------------------
# The Studio first: the dashboard compiles the renderer package from room-view's sources.
Prepare-Ui 'Sim Studio' $studio 'studio.html' @(
    (Join-Path $studio 'src'), (Join-Path $studio 'dev'), (Join-Path $studio 'vite.config.ts'), (Join-Path $studio 'package.json'))
Prepare-Ui 'dashboard' $frontend 'index.html' @(
    (Join-Path $frontend 'src'), (Join-Path $frontend 'contracts'), (Join-Path $frontend 'public'), (Join-Path $frontend 'index.html'),
    (Join-Path $frontend 'vite.config.ts'), (Join-Path $frontend 'package.json'), (Join-Path $studio 'src'))

# --- environment for the server and the producers --------------------------------------------------
$saved = @{}
function Set-Env([string]$name, [string]$value) {
    if (-not $saved.ContainsKey($name)) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    [Environment]::SetEnvironmentVariable($name, $value, 'Process')
}
Set-Env 'PYTHONPATH' "$root\backend;$root\shared\python"
Set-Env 'PYTHONUNBUFFERED' '1'
$secrets = Join-Path $root '.env.face-demo'
$face = $false
if (Test-Path -LiteralPath $secrets) {
    foreach ($line in Get-Content -LiteralPath $secrets) {
        if ($line -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$') { Set-Env $Matches[1] $Matches[2] }
    }
    Set-Env 'HEARTH_FACE_DEMO' '1'
    $face = $true
}

$children = New-Object System.Collections.ArrayList
$failed = $false
function Start-Child([string]$file, [string[]]$arguments, [string]$dir = $root) {
    $process = Start-Process -FilePath $file -ArgumentList (($arguments | ForEach-Object { Quote $_ }) -join ' ') `
        -WorkingDirectory $dir -NoNewWindow -PassThru
    $null = $process.Handle  # keeps ExitCode readable after the process ends
    [void]$children.Add($process)
    $process
}

try {
    # --- server ------------------------------------------------------------------------------------
    $bind = if ($Lan) { '0.0.0.0' } else { '127.0.0.1' }
    Write-Host "Starting the Hearth server on ${bind}:$Port..."
    $server = Start-Child $python @('-m', 'uvicorn', 'hearth.api.app:app', '--app-dir', 'backend', '--host', $bind, '--port', "$Port")
    $healthy = $false
    for ($i = 0; $i -lt 120 -and -not $healthy; $i++) {
        if ($server.HasExited) { Fail "The server stopped during startup (exit $($server.ExitCode)). See its errors above." }
        try { Invoke-RestMethod "$url/health" -TimeoutSec 1 | Out-Null; $healthy = $true } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $healthy) { Fail "The server did not answer $url/health within a minute." }

    # --- optional producers: each may fail without stopping the rest ----------------------------------
    $producers = @{}
    if ($Camera -ge 0) {
        $arguments = @('edge/run_fall_model.py', '--camera', "$Camera", '--backend', $url, '--hand-gestures', '--dashboard-preview')
        if ($face) { $arguments += '--patient-recognition' }
        $producers['webcam'] = Start-Child $python $arguments
    }
    if ($Replay) {
        $stand, $down = 'recordings/demo/stand.jsonl', 'recordings/demo/lie_down.jsonl'
        $arguments = if ($People) { @('edge/tools/replay.py', '--together', '--posture-mlp', '--loop', '--backend', $url, "$stand+$stand+$stand", "$stand+$down+$stand") }
                     else { @('edge/tools/replay.py', '--join', '--loop', '--backend', $url, $stand, $down, $stand) }
        $producers['replay'] = Start-Child $python $arguments
    }
    $vite = $null
    if ($Dev) {
        Set-Env 'HEARTH_BACKEND' $url
        $vite = Start-Child 'node' @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1') $frontend
    }

    Write-Host ''
    Write-Host "Hearth is running. Ctrl+C stops everything." -ForegroundColor Green
    Write-Host "  Dashboard       $url/          (Patient view: $url/room)"
    Write-Host "  Sim Studio      $url/studio/   Run player: $url/studio/runs"
    Write-Host "  API reference   $url/docs"
    if ($Dev) { Write-Host '  Hot reload      http://127.0.0.1:5173/' }
    Write-Host ("  Recognition     " + $(if ($face) { 'on (.env.face-demo)' } else { 'off: run scripts/face_id_demo.sh once to set it up' }))
    if ($Lan) {
        $ip = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
              Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Select-Object -First 1 -ExpandProperty IPAddress
        Write-Host "  Jetson          python3 run_edge.py --backend http://${ip}:$Port   (allow port $Port in Windows Firewall)"
    } else {
        Write-Host '  Jetson          restart with -Lan so it can reach this computer'
    }
    Write-Host ''

    $warned = @{}
    while (-not $server.HasExited) {
        foreach ($name in @($producers.Keys)) {
            if ($producers[$name].HasExited -and -not $warned[$name]) {
                $warned[$name] = $true
                Write-Host "The $name source stopped (exit $($producers[$name].ExitCode)); the rest keeps running. See its errors above." -ForegroundColor Yellow
            }
        }
        if ($vite -and $vite.HasExited -and -not $warned['vite']) {
            $warned['vite'] = $true
            Write-Host "The hot-reload server stopped; the dashboard is still at $url/." -ForegroundColor Yellow
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "The server stopped (exit $($server.ExitCode))." -ForegroundColor Yellow
    $failed = $true
} finally {
    foreach ($process in $children) {
        if (-not $process.HasExited) { & taskkill.exe /T /F /PID $process.Id 2>&1 | Out-Null }
    }
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
if ($failed) { exit 1 }
