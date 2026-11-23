# Read-only GPU / WebView2 diagnostic for the Fizgig desktop window.
# Run from the Fizgig repository root:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\diagnose_webview2.ps1
# Does not change GPU settings, kill processes, read browser cookies, or print
# full process command lines (which may contain tokens or private paths).
param([switch]$ShowAllWebViews)

$ErrorActionPreference = "Stop"
Write-Host "Fizgig WebView2 diagnostic (read only)" -ForegroundColor Cyan
Write-Host "Run while Fizgig's desktop window is open; restart it after toggling GPU."
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Host ""

try {
    $adapters = @(Get-CimInstance Win32_VideoController)
    Write-Host "Display adapters / drivers:"
    foreach ($gpu in $adapters) {
        # WMI 'Status=Error' by itself doesn't tell us if a virtual display
        # is merely disconnected, intentionally disabled, or actually broken.
        # PnP error codes distinguish those cases without modifying the device.
        $code = $gpu.ConfigManagerErrorCode
        $detail = switch ($code) {
            0  { "working" }
            10 { "cannot start" }
            22 { "disabled" }
            28 { "driver not installed" }
            31 { "driver malfunction" }
            43 { "stopped by Windows after reporting a problem" }
            45 { "not connected" }
            $null { "unavailable" }
            default { "check Device Manager" }
        }
        Write-Host ("  {0} | driver {1} | status {2} | PnP error {3} ({4})" -f
            $gpu.Name, $gpu.DriverVersion, $gpu.Status, $(if ($null -eq $code) { "N/A" } else { $code }), $detail)
    }
    if ($adapters.Count -eq 0) { Write-Host "  None reported by WMI." }
} catch {
    Write-Host ("  Could not read video controller information: " + $_.Exception.Message)
}

Write-Host ""
Write-Host "WebView2 policy (GPU acceleration):"
foreach ($key in @("HKCU:\SOFTWARE\Policies\Microsoft\Edge\WebView2",
                   "HKLM:\SOFTWARE\Policies\Microsoft\Edge\WebView2")) {
    try {
        $policy = Get-ItemProperty -LiteralPath $key -ErrorAction Stop
        if ($null -ne $policy.HardwareAccelerationModeEnabled) {
            Write-Host ("  {0}: {1}" -f $key, $policy.HardwareAccelerationModeEnabled)
        } else {
            Write-Host ("  {0}: not configured" -f $key)
        }
    } catch {
        Write-Host ("  {0}: not configured or inaccessible" -f $key)
    }
}

Write-Host ""
$flags = @(
    "--disable-gpu",
    "--disable-gpu-compositing",
    "--disable-accelerated-video-decode",
    "--disable-accelerated-2d-canvas",
    "--disable-direct-composition",
    "--use-angle",
    "--use-gl"
)
function Get-FlagSummary([string]$line) {
    $found = @()
    foreach ($flag in $flags) {
        $pattern = '(?<!\S)' + [regex]::Escape($flag) + '(?:=([^\s"]+))?(?=\s|$)'
        $match = [regex]::Match($line, $pattern)
        if ($match.Success) {
            $suffix = if ($match.Groups[1].Success) { "=" + $match.Groups[1].Value } else { "" }
            $found += $flag + $suffix
        }
    }
    if ($found.Count -eq 0) { return "(none of the tracked flags)" }
    return $found -join ", "
}

try {
    $all = @(Get-CimInstance Win32_Process -Filter "Name='msedgewebview2.exe'")
} catch {
    Write-Host "Cannot inspect running WebView2 processes: $($_.Exception.Message)"
    exit 1
}

# The pywebview app profile is separate from a standalone Chrome/Edge profile.
# Match by the app's profile name or repo name, then include child renderers.
# If nothing matches, avoid guessing which other app owns a WebView2 process.
$seeds = @($all | Where-Object {
    $_.CommandLine -match '(?i)(app_profile|[\\/]+Fizgig[\\/]+)'
})
$ids = [System.Collections.Generic.HashSet[int]]::new()
foreach ($p in $seeds) { [void]$ids.Add([int]$p.ProcessId) }
do {
    $added = $false
    foreach ($p in $all) {
        if ($ids.Contains([int]$p.ParentProcessId) -and $ids.Add([int]$p.ProcessId)) {
            $added = $true
        }
    }
} while ($added)

$chosen = if ($ShowAllWebViews) { $all } else {
    @($all | Where-Object { $ids.Contains([int]$_.ProcessId) })
}
Write-Host "Running WebView2 processes: $($all.Count) total, $($chosen.Count) shown."
if ($chosen.Count -eq 0) {
    Write-Host "No Fizgig-specific WebView2 process identified."
    Write-Host "If the desktop is open, rerun with -ShowAllWebViews to list all WebView2 processes."
} else {
    foreach ($p in ($chosen | Sort-Object ProcessId)) {
        $line = [string]$p.CommandLine
        $role = if ($line -match '(?<!\S)--type=gpu-process(?=\s|$)') {
            "gpu"
        } elseif ($line -match '(?<!\S)--type=renderer(?=\s|$)') {
            "renderer"
        } elseif ($line -match '(?<!\S)--type=utility(?=\s|$)') {
            "utility"
        } elseif ($line -match '(?<!\S)--type=') {
            "other"
        } else { "browser" }
        Write-Host ("  PID {0} | parent {1} | {2} | {3}" -f
            $p.ProcessId, $p.ParentProcessId, $role, (Get-FlagSummary $line))
    }
}

Write-Host ""
Write-Host "Check user_data\app_window.log for 'WebView2 Runtime', 'rendering',"
Write-Host "and 'event hook removal'. Compare a run with GPU ON against GPU OFF."
Write-Host "Only the app-window switch changes WebView2; normal Edge/Chrome is separate."
Write-Host "PnP code 22 (disabled) or 45 (not connected) may be expected for an unused"
Write-Host "virtual monitor; code 10, 31 or 43 warrants investigating its driver."
