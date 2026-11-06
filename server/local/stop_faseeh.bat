@echo off
cd /d "%~dp0"
docker compose --env-file settings.env down
echo.
echo  Stopped. The website goes back to the Hugging Face server by itself.
pause
