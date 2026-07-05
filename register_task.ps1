# register_task.ps1 — ŞERİT-3.2: kader-equity run_daily'i Windows Task Scheduler'a kaydeder.
# Hafta-içi (Mon-Fri), piyasa kapanışı sonrası. Saat PARAMETRİK (-Time "HH:mm", default 23:30 yerel ≈ TSİ).
# Kanonik ortam: kader-macro venv python (pyarrow gerekli). Fail-loud: run_daily nonzero exit → Task "son sonuç" != 0.
#
# Kullanım (yönetici PowerShell):
#   powershell -ExecutionPolicy Bypass -File register_task.ps1                 # default 23:30
#   powershell -ExecutionPolicy Bypass -File register_task.ps1 -Time "23:45"   # özel saat
#   powershell -ExecutionPolicy Bypass -File register_task.ps1 -Unregister     # kaldır

param(
    [string]$Time = "23:30",
    [switch]$Unregister
)

$TaskName = "KaderEquity_RunDaily"
$Py   = "C:\Users\admin\Downloads\kader-macro\.venv\Scripts\python.exe"
$Repo = "C:\Users\admin\Downloads\kader-equity"
$Script = "$Repo\run_daily.py"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "kaldırıldı: $TaskName"
    return
}

if (-not (Test-Path $Py))     { Write-Host "HATA: venv python yok: $Py (kanonik ortam gerekli)"; exit 1 }
if (-not (Test-Path $Script)) { Write-Host "HATA: run_daily.py yok: $Script"; exit 1 }

$action  = New-ScheduledTaskAction -Execute $Py -Argument "`"$Script`"" -WorkingDirectory $Repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
            -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
# StartWhenAvailable: makine kapalıyken kaçan koşuyu açılışta telafi eder (idempotent → güvenli).

# NOT (2026-06-13): run_daily artık collect_daily SONRASI / brief ÖNCESI _refresh_constan adımını
# çalıştırır — IPO boru-hattı (EDGAR S-1/F-1, günlük taze) + buyback AUTO-PULL (S&P DJI bülteni) +
# net-arz (Z.1/FRED, cache-TTL'li) + arz-talep denge (türetir) → tüm Constan bantları manuel-giriş
# olmadan canlı kalır. BEST-EFFORT/NON-FATAL: bir kaynak düşerse (FRED/EDGAR/press down) bant
# graceful-stale olur, run_daily ÖLMEZ (her fetch _substep ile sarılı, exception sızmaz; kritik
# collect+brief+ledger ayrı _step). YENİ TASK GEREKMEZ — aynı KaderEquity_RunDaily orkestratörü.
# İdempotent: çeyreklik veriler cache-TTL'li (günlük ağ-israfı yok), IPO günlük meşru taze.
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "kader-equity günlük: collect_daily + constan-refresh(IPO+buyback+net-arz+denge, best-effort) + brief(SPY/QQQ) + forward-ledger + forward-watch (fail-loud)" `
    -Force | Out-Null

Write-Host "KAYDEDİLDİ: $TaskName — hafta-içi $Time (yerel)."
Write-Host "  log: $Repo\output\run_daily.log | son sonuç: Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host "  elle test: & `"$Py`" `"$Script`""
