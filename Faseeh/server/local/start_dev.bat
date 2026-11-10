@echo off
title Faseeh TEST server (dev)
cd /d "%~dp0"
set DC=docker compose -p faseeh-dev --env-file settings.env -f docker-compose.yml -f docker-compose.dev.yml
echo.
echo  === Faseeh: starting the TEST server (GitHub branch "dev") ===
echo.
docker info >nul 2>&1
if errorlevel 1 (
  echo  Docker Desktop is not running. Open it, wait for "Engine running", then run this file again.
  pause
  exit /b 1
)
if not exist docker-compose.dev.yml (
  echo  docker-compose.dev.yml is missing: put it in this folder, next to this file.
  pause
  exit /b 1
)
docker volume inspect faseeh_cache >nul 2>&1
if errorlevel 1 (
  echo  Run start_faseeh.bat once first ^(it downloads the model^), then this file.
  pause
  exit /b 1
)
echo  Stopping the normal server ^(the website uses Hugging Face meanwhile^) ...
docker compose --env-file settings.env down >nul 2>&1
%DC% up -d --build
if errorlevel 1 (
  echo  Something went wrong. Send a photo of this window.
  pause
  exit /b 1
)
echo.
echo  Waiting for the test server ^(about 1 minute; the first time a little longer^).
set /a N=0
:wait
set STATE=
for /f %%s in ('docker inspect -f "{{.State.Health.Status}}" faseeh-dev-server-1 2^>nul') do set STATE=%%s
if "%STATE%"=="healthy" goto ready
set /a N+=1
if %N% GEQ 90 goto slow
timeout /t 10 /nobreak >nul
echo    ... still loading
goto wait
:slow
echo  It is taking unusually long. Is there a branch called dev on GitHub? Send a photo of what this shows:
%DC% logs --tail 30 server
pause
exit /b 1
:ready
set URL=
for /f "delims=" %%u in ('powershell -NoProfile -Command "$m = docker compose -p faseeh-dev logs tunnel 2>&1 | Select-String -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -Last 1; if ($m) { $m.Matches[0].Value }"') do set URL=%%u
if "%URL%"=="" (
  echo  The server is ready but the temporary address did not come. Run stop_dev.bat, then this file again.
  pause
  exit /b 1
)
echo https://faseeh.space/?team^&server=%URL%| clip
echo.
echo  ================================================================
echo   TEST SERVER READY. Open this link ^(already copied: just paste^):
echo.
echo   https://faseeh.space/?team^&server=%URL%
echo.
echo   Only a browser opened with this link uses the test server.
echo   It follows the branch "dev" by itself ^(2 minutes after an upload^).
echo   It does not save recordings. The address changes every start.
echo   When done: stop_dev.bat, then start_faseeh.bat.
echo  ================================================================
echo.
pause
