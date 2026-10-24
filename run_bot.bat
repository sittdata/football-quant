@echo off
cd /d "%~dp0"
"C:\Users\sitti\AppData\Local\Programs\Python\Python311\python.exe" telegram_bot.py >> logs\bot.log 2>&1
