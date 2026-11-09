@echo off
title Faseeh: test reciters
cd /d "%~dp0"
echo.
echo  === Faseeh: accuracy on the test reciters (Abu Bakr ash-Shatri, Maher al-Muaiqly) ===
echo  About 10 minutes. The server keeps working meanwhile. Do not close this window.
echo.
docker compose --env-file settings.env ps --status running --services 2>nul | findstr /b /c:"server" >nul
if errorlevel 1 (
  echo  The server is not running. Run start_faseeh.bat first, then this file again.
  pause
  exit /b 1
)
docker compose --env-file settings.env exec server sh -c "cd /src/repo && git fetch -q --depth 1 origin main && git reset -q --hard FETCH_HEAD; grep -q experts server/evaluate.py"
if errorlevel 1 (
  echo  The new evaluate.py is not on GitHub yet: upload it to the server folder, wait 2 minutes, then run this file again.
  pause
  exit /b 1
)
docker compose --env-file settings.env exec -e FASEEH_RESULTS=/tmp/test_results server sh -c "rm -rf /tmp/test_results /tmp/test_results.zip && python /src/repo/server/evaluate.py experts && python -m zipfile -c /tmp/test_results.zip /tmp/test_results/summary.json /tmp/test_results/test_experts.csv"
if errorlevel 1 (
  echo.
  echo  Something went wrong. Send a photo of this window.
  pause
  exit /b 1
)
docker compose --env-file settings.env cp server:/tmp/test_results.zip "%~dp0test_results.zip"
echo.
echo  Done. Send the file test_results.zip (in this folder) in the chat.
explorer /select,"%~dp0test_results.zip"
pause
