# Daily CMRL station-wise ridership fetch.
#
# The CMRL API exposes only one past day at a time, so a usable weekday sample
# can only be built by collecting once a day. src.cmrl_stationflow appends to a
# dated store and de-duplicates, so running this more than once in a day is
# harmless - and missing a day only costs that day.
#
# Registered as the scheduled task "MapathonCMRLFetch"; see scripts/README.md.

$ErrorActionPreference = "Continue"

$project = "D:\mapathon"
$logDir  = Join-Path $project "logs"
$log     = Join-Path $logDir "cmrl_fetch.log"

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $log -Value "`n===== $stamp =====" -Encoding utf8

Set-Location $project
$env:PYTHONIOENCODING = "utf-8"

# Station-wise boardings (the modelling target) plus the two system-wide
# datasets. All three share the same API limitation - one past day at a time -
# so all three only accumulate by running daily.
foreach ($module in @("src.cmrl_stationflow", "src.cmrl_extras", "src.to_excel")) {
    Add-Content -Path $log -Value "--- $module ---" -Encoding utf8
    try {
        $output = & python -m $module 2>&1
        $output | ForEach-Object { Add-Content -Path $log -Value $_ -Encoding utf8 }
        Add-Content -Path $log -Value "exit code: $LASTEXITCODE" -Encoding utf8
    } catch {
        Add-Content -Path $log -Value "FAILED: $_" -Encoding utf8
    }
}

# Keep the log from growing without bound.
if ((Test-Path $log) -and ((Get-Item $log).Length -gt 2MB)) {
    $keep = Get-Content $log -Tail 2000
    Set-Content -Path $log -Value $keep -Encoding utf8
}
