# Read-only discovery. Keep the alias so OpenSSH applies its configured user and key.
function Find-Jetson {
    [CmdletBinding()]
    param([string]$Target, [string]$LaptopIp, [switch]$Batch = $true)

    $ErrorActionPreference = 'Continue' # Windows PowerShell treats native stderr as errors.
    $targets = if ($Target) { @($Target) } else { @('jetson-wifi', 'jetson') }
    $sshOptions = @('-o', 'ConnectTimeout=5', '-o', $(if ($Batch) { 'BatchMode=yes' } else { 'BatchMode=no' }))
    foreach ($candidate in $targets) {
        # The launcher also uses the target in its tar/SSH pipeline.
        if ($candidate -notmatch '^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$') {
            Write-Warning "Invalid SSH target '$candidate'. Use an SSH config alias or user@host."
            continue
        }
        if ($Batch) {
            $output = @(& ssh.exe @sshOptions $candidate 'echo HEARTH_SSH_CONNECTION=$SSH_CONNECTION' 2>&1)
        } else {
            # Leave stderr attached to the console so SSH can show authentication prompts.
            Write-Host "Connecting to $candidate (enter your SSH password if prompted)..."
            $output = @(& ssh.exe @sshOptions $candidate 'echo HEARTH_SSH_CONNECTION=$SSH_CONNECTION')
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Cannot connect to '$candidate': $($output -join ' ')"
            continue
        }
        $connection = $output | ForEach-Object {
            if ("$_" -match '^HEARTH_SSH_CONNECTION=(\S+) \d+ (\S+) \d+$') {
                [pscustomobject]@{ LaptopIp = $Matches[1]; JetsonIp = $Matches[2] }
            }
        } | Select-Object -Last 1
        if (-not $connection) {
            Write-Warning "SSH connected to '$candidate' but did not report its network addresses."
            continue
        }
        $returnIp = if ($LaptopIp) { $LaptopIp } else { $connection.LaptopIp }
        $valid = $true
        foreach ($address in @($returnIp, $connection.JetsonIp)) {
            $parsed = $null
            if (-not [Net.IPAddress]::TryParse($address, [ref]$parsed) -or
                $parsed.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) {
                $valid = $false
            }
        }
        if (-not $valid) {
            Write-Warning "SSH connected to '$candidate', but the video relay requires IPv4 addresses. Use an IPv4 SSH host and -LaptopIp."
            continue
        }
        return [pscustomobject]@{ Target = $candidate; LaptopIp = $returnIp; JetsonIp = $connection.JetsonIp; SshOptions = $sshOptions }
    }
}
