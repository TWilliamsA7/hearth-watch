# Start Hearth: the monitoring server, the dashboard and the Sim Studio, all on one port.
#
#   .\scripts\start.ps1                 # everything: server, dashboard, Sim Studio, and the Jetson camera
#                                       #   (run_edge.py over ssh) -> http://127.0.0.1:8000/
#   .\scripts\start.ps1 -NoJetson       # ...without the Jetson (simulation and replay still work)
#   .\scripts\start.ps1 -Camera 0       # ...plus this laptop's webcam as another live camera
#   .\scripts\start.ps1 -Replay         # ...plus recorded clips replayed as replay-01
#   .\scripts\start.ps1 -Replay -People # ...the same clips as two people at once (one lies down)
#   .\scripts\start.ps1 -Dev            # ...plus a Vite dev server with hot reload on :5173
#
# It installs the dashboard's packages and rebuilds it when its sources changed (-NoBuild skips
# that), and installs face recognition's packages and models when they are missing. The Jetson is
# the ssh host "jetson" (-Jetson to change); it reaches this laptop at 192.168.55.100 over USB
# (-LaptopIp to change). Its face-ID token is read from the Jetson. Recognition's key is kept in
# data/face.key. Everything runs in this window; Ctrl+C stops all of it, the Jetson's camera too.
param(
    [int]$Port = 8000,
    [switch]$NoJetson,
    [string]$Jetson = 'jetson',
    [string]$LaptopIp = '192.168.55.100',
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

# --- face recognition: optional, so problems here only warn -----------------------------------------
$ErrorActionPreference = 'Continue'
Push-Location $root
& $python -c 'import cryptography, cv2, numpy, PIL' 2>$null
if ($LASTEXITCODE) {
    Write-Host 'Installing face recognition packages...'
    & $python -m pip install --disable-pip-version-check -q -e "$root\backend[face-demo]"
    if ($LASTEXITCODE) { Write-Host 'Could not install the face recognition packages; everything else will run.' -ForegroundColor Yellow }
}
# The Jetson's video reaches the dashboard through a relay that decodes it with PyAV.
& $python -c 'import av' 2>$null
if ($LASTEXITCODE -and -not $NoJetson) {
    Write-Host 'Installing the Jetson video relay package...'
    & $python -m pip install --disable-pip-version-check -q -e "$root\backend[jetson]"
    if ($LASTEXITCODE) { Write-Host 'Could not install PyAV; the Jetson video will not show (its skeletons still will).' -ForegroundColor Yellow }
}
& $python -c 'import sys; sys.path.insert(0, ''backend''); from hearth.perception.face_demo import MODELS, DEFAULT_DIR, verify; [verify(DEFAULT_DIR / p.split(''/'')[1], h) for p, h in MODELS.items()]' 2>$null
if ($LASTEXITCODE) {
    Write-Host 'Downloading the face recognition models...'
    & $python -c 'import sys; sys.path.insert(0, ''backend''); from hearth.perception.face_demo import download; download()'
    if ($LASTEXITCODE) { Write-Host 'Could not download the face recognition models; everything else will run.' -ForegroundColor Yellow }
}
Pop-Location

# --- laptop webcam (optional): download its models once, so the camera client can start ------------
$NoCamera = $Camera -lt 0
if (-not $NoCamera) {
    & $python -c 'import ultralytics, torch' 2>$null
    if ($LASTEXITCODE) {
        Write-Host "The webcam needs PyTorch and Ultralytics; starting without it. Install once with:" -ForegroundColor Yellow
        Write-Host "  .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor Yellow
        Write-Host "  .\.venv\Scripts\python.exe -m pip install -e './backend[vision]'" -ForegroundColor Yellow
        $NoCamera = $true
    } elseif (-not (Test-Path -LiteralPath (Join-Path $root 'models\human-fall-detection\best.pt')) -or
              -not (Test-Path -LiteralPath (Join-Path $root 'models\hand-gesture\gesture.pt'))) {
        Write-Host 'Downloading the webcam models...'
        Push-Location $root
        & $python edge/run_fall_model.py --download-only --hand-gestures
        Pop-Location
        if ($LASTEXITCODE) { Write-Host 'Could not download the webcam models; starting without the webcam.' -ForegroundColor Yellow; $NoCamera = $true }
    }
}

# --- Jetson: reachable over ssh? (problems here only warn; everything else still runs) -------------
$ssh = @('-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes', $Jetson)
$useJetson = $false
$jetsonToken = $null
if (-not $NoJetson) {
    & ssh.exe @ssh true 2>$null
    if ($LASTEXITCODE) {
        Write-Host "Cannot reach the Jetson as '$Jetson' (USB cable in? powered on? try: ssh $Jetson). Starting without it." -ForegroundColor Yellow
    } elseif (& ssh.exe @ssh 'pgrep -f ''^python3 -u run_edge.py'' >/dev/null && echo busy') {
        Write-Host "run_edge.py is already running on the Jetson; leaving it alone (stop it with: ssh $Jetson pkill -INT -f run_edge.py)." -ForegroundColor Yellow
    } else {
        $useJetson = $true
        # The Jetson's face-ID token, so the server accepts its face templates (never printed).
        $jetsonToken = (& ssh.exe @ssh 'cat ~/.hearth-face-token 2>/dev/null') | Select-Object -First 1
    }
}
$ErrorActionPreference = 'Stop'
$running = $null
try { $running = Invoke-RestMethod "$url/health" -TimeoutSec 2 } catch {}
if ($running) { Fail "Hearth is already running on port $Port ($url/). Stop it first (Ctrl+C in its window), or use -Port." }
# Windows lets 127.0.0.1 and 0.0.0.0 bind the same port separately, so check both.
foreach ($address in @([Net.IPAddress]::Any, [Net.IPAddress]::Loopback)) {
    try { $probe = [Net.Sockets.TcpListener]::new($address, $Port); $probe.Start(); $probe.Stop() }
    catch { Fail "Port $Port is in use by another program. Free it or choose another with -Port." }
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
# An existing face key and Jetson token (from scripts/face_id_demo.sh) keep working.
$secrets = Join-Path $root '.env.face-demo'
if (Test-Path -LiteralPath $secrets) {
    foreach ($line in Get-Content -LiteralPath $secrets) {
        if ($line -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$') { Set-Env $Matches[1] $Matches[2] }
    }
}
if ($jetsonToken -and -not $env:HEARTH_EDGE_FACE_TOKEN) { Set-Env 'HEARTH_EDGE_FACE_TOKEN' $jetsonToken.Trim() }

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
    # The Jetson posts over the USB network, so listen beyond this computer when it is used.
    $bind = if ($Lan -or $useJetson) { '0.0.0.0' } else { '127.0.0.1' }
    Write-Host "Starting the Hearth server on ${bind}:$Port..."
    # No per-request access log: the cameras post many times a second and would bury real messages.
    $server = Start-Child $python @('-m', 'uvicorn', 'hearth.api.app:app', '--app-dir', 'backend', '--host', $bind, '--port', "$Port", '--no-access-log')
    $healthy = $false
    for ($i = 0; $i -lt 120 -and -not $healthy; $i++) {
        if ($server.HasExited) { Fail "The server stopped during startup (exit $($server.ExitCode)). See its errors above." }
        try { Invoke-RestMethod "$url/health" -TimeoutSec 1 | Out-Null; $healthy = $true } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $healthy) { Fail "The server did not answer $url/health within a minute." }
    $recognition = $null
    try { $recognition = Invoke-RestMethod "$url/face-demo/status" -TimeoutSec 5 } catch {}
    $face = [bool]($recognition -and $recognition.enabled)

    # --- optional producers: each may fail without stopping the rest ----------------------------------
    $producers = @{}
    $jetsonStarted = $false
    if ($useJetson) {
        $ErrorActionPreference = 'Continue'
        # Latest edge/ code onto the Jetson (tar over ssh, swapped in whole; CRLF stripped there).
        $copy = "cd ~ && mkdir -p hearth-watch && cd hearth-watch && rm -rf edge.new && mkdir edge.new && tar -xf - -C edge.new --strip-components=1 && find edge.new \( -name '*.sh' -o -name '*.py' \) -exec sed -i 's/\r`$//' {} + && rm -rf edge && mv edge.new edge"
        & cmd.exe /c "tar -C `"$root`" --exclude=__pycache__ --exclude=tests -cf - edge | ssh -o BatchMode=yes $Jetson `"$copy`""
        if ($LASTEXITCODE) { Write-Host 'Could not copy edge/ to the Jetson; using the code already there.' -ForegroundColor Yellow }
        # The backend drops observations more than 3 s old, so the clocks must agree.
        $before = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000
        $remote = (& ssh.exe @ssh 'date +%s.%N') | Select-Object -First 1
        $after = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000
        $offset = [double]$remote - ($before + $after) / 2
        if ([Math]::Abs($offset) -gt 1) {
            Write-Host ("The Jetson clock is {0:N1} s off this laptop's; its observations will be dropped as stale. Fix: edge/tools/sync_clock.sh" -f $offset) -ForegroundColor Yellow
        }
        $edgeUrl = "http://${LaptopIp}:$Port"
        & ssh.exe @ssh "curl -s -m 3 $edgeUrl/health >/dev/null"
        if ($LASTEXITCODE) {
            Write-Host "The Jetson cannot reach this server at $edgeUrl (allow Python through Windows Firewall on private networks, or pass -LaptopIp)." -ForegroundColor Yellow
        }
        # Face ID when the Jetson has its token and engine and this server has the same token.
        $faceId = ''
        if ($env:HEARTH_EDGE_FACE_TOKEN -and (& ssh.exe @ssh 'test -f ~/.hearth-face-token -a -f ~/hearth-watch/models/face-id/sface_fp16.engine && echo yes')) {
            $faceId = '--face-id --face-id-token-file ~/.hearth-face-token'
        }
        # -tt: Ctrl+C reaches run_edge.py so it closes the camera cleanly. Only our [edge] lines are shown.
        # The clean camera image streams to this laptop; the relay hands it to the dashboard, which
        # draws the skeletons and posture calls from the observations.
        $producers['Jetson video relay'] = Start-Child $python @('-m', 'hearth.perception.stream_relay', '--backend', $url, '--laptop-ip', $LaptopIp)
        $edgeCommand = "cd ~/hearth-watch/edge && python3 -u run_edge.py --backend $edgeUrl $faceId --stream rtp://${LaptopIp}:1234 --stream-plain 2>&1 >/dev/null | (trap '' INT; grep --line-buffered -aE '\[edge')"
        $producers['Jetson camera'] = Start-Child 'ssh.exe' @('-tt', '-o', 'BatchMode=yes', $Jetson, $edgeCommand)
        $jetsonStarted = $true
        $ErrorActionPreference = 'Stop'
    }
    if (-not $NoCamera) {
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
    Write-Host "  Dashboard       $url/          (Scenarios: $url/scenarios, Recognition: $url/recognition)"
    Write-Host "  Sim Studio      $url/studio/   Run player: $url/studio/runs"
    Write-Host "  API reference   $url/docs"
    if ($Dev) { Write-Host '  Hot reload      http://127.0.0.1:5173/' }
    Write-Host ("  Recognition     " + $(if ($face) { 'on' } elseif ($recognition) { "off: $($recognition.message)" } else { 'status unavailable' }))
    Write-Host ("  Jetson camera   " + $(if ($jetsonStarted) { "running ($Jetson -> http://${LaptopIp}:$Port$(if ($faceId) { ', face ID on' }))" } elseif ($NoJetson) { 'off (-NoJetson)' } else { 'not started (see above)' }))
    if (-not $NoCamera) { Write-Host "  Laptop webcam   camera $Camera" }
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
    # Stop the Jetson program this script started (never someone else's run), freeing its camera.
    if ($jetsonStarted) { & ssh.exe -o ConnectTimeout=5 -o BatchMode=yes $Jetson 'pkill -INT -f ''^python3 -u run_edge.py''' 2>&1 | Out-Null }
    foreach ($process in $children) {
        if (-not $process.HasExited) { & taskkill.exe /T /F /PID $process.Id 2>&1 | Out-Null }
    }
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
if ($failed) { exit 1 }
