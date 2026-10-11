# Run with: powershell -NoProfile -File scripts/test-jetson.ps1
# Mock SSH so these checks never connect to or change a device.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'jetson.ps1')
function Assert($condition, [string]$message) {
    if (-not $condition) { throw $message }
}
function ssh.exe {
    $target = $args[4]
    $script:calls += $target
    Assert ($args[1] -eq 'ConnectTimeout=5' -and $args[3] -eq $script:expectedMode) 'SSH must use the selected authentication mode'
    Assert ($args[5] -eq 'echo HEARTH_SSH_CONNECTION=$SSH_CONNECTION') 'Probe must ask the remote shell for its addresses'
    $reply = $script:replies[$target]
    if ($null -eq $reply) {
        $global:LASTEXITCODE = 255
        'Permission denied (publickey).'
    } else {
        $global:LASTEXITCODE = 0
        $reply
    }
}

$script:calls = @()
$script:expectedMode = 'BatchMode=yes'
$script:replies = @{ 'jetson-wifi' = @('Welcome', 'HEARTH_SSH_CONNECTION=100.90.80.70 50000 100.76.117.50 22') }
$found = Find-Jetson
Assert ($found.Target -eq 'jetson-wifi' -and $calls.Count -eq 1) 'Prefer the Tailscale alias'
Assert ($found.LaptopIp -eq '100.90.80.70' -and $found.JetsonIp -eq '100.76.117.50') 'Use both Tailscale endpoints'
Assert ($found.SshOptions[3] -eq 'BatchMode=yes') 'Default startup must not repeatedly prompt for a password'

$script:expectedMode = 'BatchMode=yes'
$found = Find-Jetson -Batch
Assert ($found.Target -eq 'jetson-wifi' -and $found.SshOptions[3] -eq 'BatchMode=yes') 'Support unattended key authentication'
$script:expectedMode = 'BatchMode=no'
$found = Find-Jetson -Batch:$false
Assert ($found.SshOptions[3] -eq 'BatchMode=no') 'Allow explicitly requested password authentication'
$script:expectedMode = 'BatchMode=yes'

$script:calls = @()
$script:replies = @{ jetson = 'HEARTH_SSH_CONNECTION=192.168.55.100 50000 192.168.55.1 22' }
$found = Find-Jetson -WarningVariable warnings -WarningAction SilentlyContinue
Assert ($found.Target -eq 'jetson' -and ($calls -join ',') -eq 'jetson-wifi,jetson') 'Fall back to USB'
Assert ($found.LaptopIp -eq '192.168.55.100') 'Detect USB return address'
Assert ("$warnings" -match 'Permission denied') 'Preserve SSH diagnostics'

$script:calls = @()
$found = Find-Jetson -Target 'jetson-wifi' -WarningAction SilentlyContinue
Assert ($null -eq $found -and $calls.Count -eq 1) 'Explicit target must not fall back to another device'

$script:replies = @{ custom = 'HEARTH_SSH_CONNECTION=100.90.80.70 50000 100.76.117.50 22' }
$found = Find-Jetson -Target custom -LaptopIp '10.1.2.3'
Assert ($found.Target -eq 'custom' -and $found.LaptopIp -eq '10.1.2.3') 'Honor explicit target and return address'
$found = Find-Jetson -Target custom -LaptopIp 'invalid' -WarningAction SilentlyContinue
Assert ($null -eq $found) 'Reject invalid return addresses'

foreach ($reply in @('Welcome only', 'HEARTH_SSH_CONNECTION=::1 50000 ::1 22')) {
    $script:replies = @{ custom = $reply }
    $found = Find-Jetson -Target custom -WarningAction SilentlyContinue
    Assert ($null -eq $found) 'Do not silently use USB defaults for missing or unsupported addresses'
}
$script:calls = @()
$found = Find-Jetson -Target 'bad&target' -WarningAction SilentlyContinue
Assert ($null -eq $found -and $calls.Count -eq 0) 'Reject shell metacharacters in targets'

# Parse the launcher without executing its installs, sync or camera startup.
$tokens = $null
$parseErrors = $null
$null = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'start.ps1'), [ref]$tokens, [ref]$parseErrors)
Assert ($parseErrors.Count -eq 0) "Launcher syntax errors: $parseErrors"
Write-Host 'Jetson discovery checks passed.'
