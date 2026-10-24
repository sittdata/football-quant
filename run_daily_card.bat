@echo off
:: Daily Quant Betting Card - Auto Push to Telegram
:: Scheduled to run at 17:30 via Windows Task Scheduler

cd /d "C:\Users\sitti\OneDrive\Desktop\football"
python scripts/daily_telegram_card.py >> logs\daily_card.log 2>&1
