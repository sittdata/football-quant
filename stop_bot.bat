@echo off
echo Stopping FBPredict Telegram Bot...
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*telegram_bot.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host 'Stopped bot process:' $_.ProcessId }"
echo Done.
pause
