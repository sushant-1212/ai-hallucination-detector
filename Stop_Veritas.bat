@echo off
echo Stopping Veritas API background process...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do taskkill /f /pid %%a >nul 2>&1
echo Done! Veritas API has been stopped.
pause
