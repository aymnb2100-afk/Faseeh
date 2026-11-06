@echo off
cd /d "%~dp0"
echo  This deletes the Faseeh server from this computer: its containers, the downloaded model and the image.
choice /m "  Delete everything"
if errorlevel 2 exit /b 0
docker compose --env-file settings.env down -v --rmi all
echo.
echo  Done. Nothing of the server is left. You can now delete this folder too.
pause
