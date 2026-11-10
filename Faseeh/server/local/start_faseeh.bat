@echo off
title Faseeh server
cd /d "%~dp0"
echo.
echo  === Faseeh: starting the analysis server on this computer ===
echo.
docker info >nul 2>&1
if errorlevel 1 (
  echo  Docker Desktop is not running.
  echo  Open Docker Desktop, wait until it shows "Engine running", then run this file again.
  pause
  exit /b 1
)
findstr /c:"PUT_YOUR" settings.env >nul
if not errorlevel 1 (
  echo  First put your ngrok details in settings.env ^(it opens now^), save it, then run this file again.
  notepad settings.env
  pause
  exit /b 1
)
rem the test server (start_dev.bat) and this one never run together: one graphics card
if exist docker-compose.dev.yml docker compose -p faseeh-dev --env-file settings.env -f docker-compose.yml -f docker-compose.dev.yml down >nul 2>&1
docker compose --env-file settings.env up -d --build
if errorlevel 1 (
  echo  Something went wrong. Send a photo of this window.
  pause
  exit /b 1
)
for /f "tokens=2 delims==" %%d in ('findstr /b NGROK_DOMAIN settings.env') do set DOMAIN=%%d
echo.
echo  Waiting for the server. The very first time it downloads the model ^(about 2.5 GB^), later about 1 minute.
set /a N=0
:wait
set STATE=
for /f %%s in ('docker inspect -f "{{.State.Health.Status}}" faseeh-server-1 2^>nul') do set STATE=%%s
if "%STATE%"=="healthy" goto ready
set /a N+=1
if %N% GEQ 120 goto slow
timeout /t 10 /nobreak >nul
echo    ... still loading
goto wait
:slow
echo  It is taking unusually long. Run logs_faseeh.bat and send a photo of what it shows.
pause
exit /b 1
:ready
echo.
docker compose --env-file settings.env exec -T server python -c "import torch; print('  Graphics card:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND (running on the processor)')"
echo.
echo  ================================================================
echo   READY.  The website uses this computer by itself:  faseeh.space
echo   Check:  https://%DOMAIN%/health
echo   It keeps itself updated from GitHub while it runs.
echo   It stays on until you run stop_faseeh.bat (or turn off the
echo   computer); it never starts by itself. You can close this window.
echo  ================================================================
echo.
pause
