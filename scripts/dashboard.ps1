# Run from the repository root. Only 5173 is the user-facing dashboard.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Install the repository .venv first.' }
$env:PYTHONPATH = "$root\backend;$root\shared\python"
$backend = $null
try {
    $schema = $null
    try { $schema = Invoke-RestMethod http://127.0.0.1:8002/openapi.json -TimeoutSec 2 } catch {}
    if (-not $schema) {
        $backend = Start-Process $python -WorkingDirectory $root -WindowStyle Hidden -PassThru -ArgumentList @('-m','uvicorn','hearth.api.app:app','--host','127.0.0.1','--port','8002')
        for ($i=0; $i -lt 30; $i++) {
            if ($backend.HasExited) { throw 'Backend failed to start. Check dependencies and port 8002.' }
            try { $schema = Invoke-RestMethod http://127.0.0.1:8002/openapi.json -TimeoutSec 1; break } catch { Start-Sleep -Milliseconds 500 }
        }
    }
    foreach ($route in @('/appointments','/face-demo/status','/sim/live/status')) {
        if (-not $schema -or -not $schema.paths.PSObject.Properties[$route]) { throw "Stop the old API process on 8002 and retry: missing $route." }
    }
    Write-Host 'Open http://127.0.0.1:5173/ — the main dashboard.'
    Push-Location (Join-Path $root 'frontend')
    try { & npm.cmd run dev } finally { Pop-Location }
} finally {
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id }
}
