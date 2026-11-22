@echo off
title Faseeh: test on the website recordings
cd /d "%~dp0"
echo.
echo  === Faseeh: the reviewed website recordings, analysed again with today's settings and timed ===
echo  A few minutes (about 5 to 10 seconds per recording). The server keeps working meanwhile. Do not close this window.
echo.
docker compose --env-file settings.env ps --status running --services 2>nul | findstr /b /c:"server" >nul
if errorlevel 1 (
  echo  The server is not running. Run start_faseeh.bat first, then this file again.
  pause
  exit /b 1
)
docker compose --env-file settings.env exec server sh -c "cd /src/repo && git fetch -q --depth 1 origin main && git reset -q --hard FETCH_HEAD; grep -q def.site server/evaluate.py"
if errorlevel 1 (
  echo  The new evaluate.py is not on GitHub yet: upload it to the server folder, wait 2 minutes, then run this file again.
  pause
  exit /b 1
)
docker compose --env-file settings.env exec -e FASEEH_RESULTS=/tmp/site_results server sh -c "rm -rf /tmp/site_results /tmp/site_results.zip && python /src/repo/server/evaluate.py site && python -m zipfile -c /tmp/site_results.zip /tmp/site_results/summary.json /tmp/site_results/site.csv"
if errorlevel 1 (
  echo.
  echo  Something went wrong. Send a photo of this window.
  pause
  exit /b 1
)
docker compose --env-file settings.env cp server:/tmp/site_results.zip "%~dp0site_results.zip"
echo.
echo  Done. Send the file site_results.zip (in this folder) in the chat.
explorer /select,"%~dp0site_results.zip"
pause
