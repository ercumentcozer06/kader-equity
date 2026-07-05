# register_preopen.ps1 — kader-equity'i ABD açılışından 1 dk sonra (09:31 ET) çalıştıran görevi kaydeder.
#
# DST mantığı: Türkiye sabit UTC+3; ABD-Doğu DST ile kayar → 09:31 ET = YAZ 16:31 TSİ / KIŞ 17:31 TSİ.
# Çözüm: İKİ tetik (16:31 + 17:31 TSİ, hafta-içi). Python kapısı (run_preopen.py) gerçek ABD/Doğu saatini
# hesaplar; yalnız 09:31 ET firing'i run_daily'i koşar, diğeri sessizce atlar → DST otomatik, elle saat YOK.
#
#   powershell -ExecutionPolicy Bypass -File register_preopen.ps1               # kaydet
#   powershell -ExecutionPolicy Bypass -File register_preopen.ps1 -Unregister   # kaldır
param([switch]$Unregister)

$ErrorActionPreference = "Stop"
$Repo     = "C:\Users\admin\Downloads\kader-equity"
$Py       = "C:\Users\admin\Downloads\kader-macro\.venv\Scripts\python.exe"
$Script   = "$Repo\run_preopen.py"
$TaskName = "KaderEquity_PreOpen"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "kaldırıldı: $TaskName"
    return
}
if (-not (Test-Path $Py))     { Write-Host "HATA: python yok: $Py";          exit 1 }
if (-not (Test-Path $Script)) { Write-Host "HATA: run_preopen.py yok: $Script"; exit 1 }

# Eski 23:30 post-close görevi varsa kaldır (çift-koşmayı önle; pre-open onun yerine geçer).
Unregister-ScheduledTask -TaskName "KaderEquity_RunDaily" -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction -Execute $Py -Argument "`"$Script`"" -WorkingDirectory $Repo
$days   = @("Monday","Tuesday","Wednesday","Thursday","Friday")
$t1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At "16:31"   # YAZ (EDT): 09:31 ET = açılış+1dk
$t2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At "17:31"   # KIŞ (EST): 09:31 ET = açılış+1dk
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $t1, $t2 -Settings $settings `
    -Description "kader-equity: ABD acilistan 1dk sonra (09:31 ET) canli gamma/greek + model kos. Iki tetik 16:31/17:31 TSI, run_preopen.py kapisi yalniz 09:31 ET kosar (DST-otomatik)." -Force | Out-Null

Write-Host "KAYDEDILDI: $TaskName"
Write-Host "  tetikler : hafta-ici 16:31 + 17:31 TSI (yalniz 09:31 ET kosar; DST-otomatik)"
$i = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Host "  NextRun  : $($i.NextRunTime)"
