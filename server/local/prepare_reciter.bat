@echo off
title Faseeh: new reciter
cd /d "%~dp0"
echo.
echo  === Faseeh: preparing the new reciter (al-Minshawi, murattal) ===
echo  About half an hour. The server keeps working meanwhile. Do not close this window.
echo.
docker compose --env-file settings.env ps --status running --services 2>nul | findstr /b /c:"server" >nul
if errorlevel 1 (
  echo  The server is not running. Run start_faseeh.bat first, then this file again.
  pause
  exit /b 1
)
docker compose --env-file settings.env exec server sh -c "cd /src/repo && git fetch -q --depth 1 origin main && git reset -q --hard FETCH_HEAD; test -f server/prepare_reciter.py"
if errorlevel 1 (
  echo  prepare_reciter.py is not on GitHub yet: upload it to the server folder, wait 2 minutes, then run this file again.
  pause
  exit /b 1
)
docker compose --env-file settings.env exec server python /src/repo/server/prepare_reciter.py
if errorlevel 1 (
  echo.
  echo  Something went wrong. Send a photo of this window.
  pause
  exit /b 1
)
docker compose --env-file settings.env cp server:/tmp/reciter_out.zip "%~dp0reciter_out.zip"
echo.
echo  Done. Send the file reciter_out.zip (in this folder) in the chat.
explorer /select,"%~dp0reciter_out.zip"
pause
