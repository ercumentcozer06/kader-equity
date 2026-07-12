# register_heartbeat.ps1 — kader-equity DEAD-MAN heartbeat'ini zamanlayıcıya kaydeder.
# run_daily HİÇ koşmazsa (çökme/Scheduler-off/makine-kapalı) yakalar → notify.alert push.
# Günde 3× (09:00/13:00/18:00 TSİ); heartbeat.py içi 36s-sessizlik toleransı hafta-sonu/tatili yutar.
# S4U dener (oturum kapalıyken de koşar, watchdog kardeşiyle aynı desen); elevation gerekirse
# default-logon'a düşer (makine hep açık/oturumlu → fonksiyonel fark yok).
#
#   powershell -ExecutionPolicy Bypass -File register_heartbeat.ps1              # kaydet
#   powershell -ExecutionPolicy Bypass -File register_heartbeat.ps1 -Unregister  # kaldır
param([switch]$Unregister)

$ErrorActionPreference = "Stop"
$Repo     = "C:\Users\admin\Downloads\kader-equity"
$Py       = "C:\Users\admin\Downloads\kader-macro\.venv\Scripts\python.exe"
$Script   = "$Repo\heartbeat.py"
$TaskName = "KaderEquity_Heartbeat"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "kaldırıldı: $TaskName"
    return
}
if (-not (Test-Path $Py))     { Write-Host "HATA: python yok: $Py";       exit 1 }
if (-not (Test-Path $Script)) { Write-Host "HATA: heartbeat.py yok: $Script"; exit 1 }

$action  = New-ScheduledTaskAction -Execute $Py -Argument "`"$Script`"" -WorkingDirectory $Repo
$trigs   = @("09:00","13:00","18:00") | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ }
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
$desc = "kader-equity DEAD-MAN: run_daily 36s+ sessizse push-alarm. 3x/gun 09/13/18 TSI. Heartbeat.py piyasa-kapali gunleri hos gorur."

$mode = "S4U"
try {
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigs -Principal $principal `
        -Settings $settings -Description $desc -Force -ErrorAction Stop | Out-Null
} catch {
    $mode = "DEFAULT (S4U elevation gerekti -> interactive-user'a dusuldu; admin shell'de -Force ile S4U'ya yukseltilebilir)"
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigs `
        -Settings $settings -Description $desc -Force | Out-Null
}

Write-Host "KAYDEDILDI: $TaskName  [$mode]"
Write-Host "  tetikler : gunluk 09:00 / 13:00 / 18:00 TSI"
$i = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Host "  NextRun  : $($i.NextRunTime)"
