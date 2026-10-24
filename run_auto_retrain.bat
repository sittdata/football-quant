@echo off
:: Automated Multi-League Model Retraining Pipeline
:: Ingests results, benchmarks candidate vs production, promotes model, and notifies Telegram

cd /d "C:\Users\sitti\OneDrive\Desktop\football"
python scripts\auto_retrain_pipeline.py >> logs\auto_retrain.log 2>&1
