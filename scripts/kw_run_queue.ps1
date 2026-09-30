# Unattended run queue for the keyword study: generate -> score for each model, then analyse.
#
#   powershell -ExecutionPolicy Bypass -File scripts\kw_run_queue.ps1 -Models qwen2.5-7b,mistral-7b -WaitPid 12152
#   (-WaitPid takes several PIDs, e.g. an earlier queue and a weight download: -WaitPid 74440,46168)
#
# Every step runs offline (HF_HUB_OFFLINE=1) from local caches, so a flaky network cannot
# stall an overnight run. 7B weights live in D:\hf_cache and are selected per step through
# HF_HUB_CACHE; scoring models stay in the default cache. Each step logs to results\kw\.
# All steps are resumable, so re-running the queue after an interruption continues it.

param(
    [string[]]$Models = @("qwen2.5-7b", "mistral-7b"),
    [int[]]$WaitPid = @(),
    [string]$Config = "configs/kw_week1.toml",
    [string]$BigModelCache = "D:\hf_cache",
    # Small models whose weights are in the default C: cache rather than $BigModelCache.
    [string[]]$DefaultCacheModels = @("llama3.2-1b", "llama3.2-3b")
)

$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\nisha\miniconda3\python.exe"
$logs = Join-Path $root "results\kw"
$queueLog = Join-Path $logs "queue.log"

function Note($msg) { "$(Get-Date -Format s)  $msg" | Tee-Object -FilePath $queueLog -Append }

function Step($name, [string[]]$pyArgs, [bool]$bigCache) {
    $env:HF_HUB_OFFLINE = "1"
    if ($bigCache) { $env:HF_HUB_CACHE = $BigModelCache } else { [Environment]::SetEnvironmentVariable("HF_HUB_CACHE", $null, "Process") }
    Note "start $name"
    $p = Start-Process -FilePath $python -ArgumentList (@("-u") + $pyArgs) -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $logs "$name.log") -RedirectStandardError (Join-Path $logs "$name.log.err") `
        -WindowStyle Hidden -PassThru -Wait
    Note "end $name (exit $($p.ExitCode))"
}

foreach ($waitFor in $WaitPid) {
    Note "waiting for PID $waitFor"
    while (Get-Process -Id $waitFor -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}

# Named after this queue's first model, so a chained queue keeps the earlier queue's log.
Step "score_before_$($Models[0])" @("scripts/kw_score.py", "--config", $Config) $false
foreach ($m in $Models) {
    Step "run_$m" @("scripts/kw_generate.py", "--config", $Config, "--model", $m, "--split", "test") ($m -notin $DefaultCacheModels)
    Step "score_$m" @("scripts/kw_score.py", "--config", $Config) $false
}
Step "analyze" @("scripts/kw_analyze.py", "--config", $Config) $false
Note "queue finished"
