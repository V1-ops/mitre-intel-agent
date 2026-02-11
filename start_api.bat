@echo off
echo ========================================
echo Starting MITRE SOC Copilot API
echo ========================================
echo.
echo Loading models... (this takes 10-15 seconds)
echo.
uvicorn test_api:app --reload
