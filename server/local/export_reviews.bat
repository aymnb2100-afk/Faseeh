@echo off
title Faseeh: export the reviews
cd /d "%~dp0"
echo.
echo  === Faseeh: packing the recordings and the reviews ===
echo.
docker compose --env-file settings.env ps --status running --services 2>nul | findstr /b /c:"server" >nul
if errorlevel 1 (
  echo  The server is not running. Run start_faseeh.bat first, then this file again.
  pause
  exit /b 1
)
docker compose --env-file settings.env exec server sh -c "cd /src/repo && git fetch -q --depth 1 origin main && git reset -q --hard FETCH_HEAD; test -f server/export_reviews.py"
if errorlevel 1 (
  echo  export_reviews.py is not on GitHub yet: upload it to the server folder, wait 2 minutes, then run this file again.
  pause
  exit /b 1
)
docker compose --env-file settings.env exec server sh -c "rm -rf /tmp/reviews /tmp/reviews.zip && python /src/repo/server/export_reviews.py"
if errorlevel 1 (
  echo.
  echo  Something went wrong. Send a photo of this window.
  pause
  exit /b 1
)
docker compose --env-file settings.env cp server:/tmp/reviews.zip "%~dp0reviews.zip"
echo.
echo  Done. Send the file reviews.zip (in this folder) in the chat.
explorer /select,"%~dp0reviews.zip"
pause
