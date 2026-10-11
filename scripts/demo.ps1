# Start the Hearth demo on this laptop: backend (monitor, nurse desk, simulator) and the Sim Studio.
#
#   .\scripts\demo.ps1                 # backend on 0.0.0.0:8000 + Studio on :5174
#   .\scripts\demo.ps1 -Camera 0       # ...plus the YOLO fall model on webcam 0, with a preview window
#   .\scripts\demo.ps1 -Replay         # ...plus Gaston's recorded clips replayed as replay-01 (no Jetson needed)
#   .\scripts\demo.ps1 -Replay -People # ...the same clips as two people at once (p2 lies down), with our posture model's calls
#
# Each part runs in its own window; close a window (or Ctrl+C in it) to stop that part.
param(
    [int]$Camera = -1,
    [switch]$Replay,
    [switch]$People,
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "No .venv: python -m venv .venv; .venv\Scripts\python.exe -m pip install -e ./shared/python -e './backend[dev]'" }

Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command",
    "& '$python' -m uvicorn hearth.api.app:app --host 0.0.0.0 --port $Port"

$roomView = Join-Path $root "frontend\room-view"
if (-not (Test-Path (Join-Path $roomView "node_modules"))) { Push-Location $roomView; npm install; Pop-Location }
Start-Process powershell -WorkingDirectory $roomView -ArgumentList "-NoExit", "-Command",
    "`$env:HEARTH_BACKEND='http://127.0.0.1:$Port'; npm run dev"

# Wait for the backend before starting anything that sends to it.
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-RestMethod "http://127.0.0.1:$Port/health" | Out-Null; break } catch { Start-Sleep -Milliseconds 500 }
}

if ($Camera -ge 0) {
    Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command",
        "& '$python' edge/run_fall_model.py --camera $Camera --backend http://127.0.0.1:$Port --preview"
}
if ($Replay -and $People) {
    $stand, $down = "recordings/demo/stand.jsonl", "recordings/demo/lie_down.jsonl"
    Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command",
        "& '$python' edge/tools/replay.py --together --posture-mlp --loop --backend http://127.0.0.1:$Port $stand+$stand+$stand $stand+$down+$stand"
} elseif ($Replay) {
    $clips = "recordings/demo/stand.jsonl", "recordings/demo/lie_down.jsonl", "recordings/demo/stand.jsonl"
    Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command",
        "& '$python' edge/tools/replay.py --join --loop --backend http://127.0.0.1:$Port $($clips -join ' ')"
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
       Select-Object -First 1 -ExpandProperty IPAddress)
Write-Host ""
Write-Host "Sim Studio   http://localhost:5174/studio.html   (Live camera panel: Link into this room)"
Write-Host "Nurse desk   http://127.0.0.1:$Port/nurses"
Write-Host "Room button  http://127.0.0.1:$Port/room"
Write-Host ""
Write-Host "On the Jetson:  python3 run_edge.py --backend http://${ip}:$Port   (sync the clock first: edge/tools/sync_clock.sh)"
Write-Host "Without it:     .\scripts\demo.ps1 -Replay   (then link source replay-01 in the Studio)"
