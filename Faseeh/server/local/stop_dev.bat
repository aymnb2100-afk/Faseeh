@echo off
cd /d "%~dp0"
docker compose -p faseeh-dev --env-file settings.env -f docker-compose.yml -f docker-compose.dev.yml down
echo.
echo  The test server is stopped. Run start_faseeh.bat to bring the normal server back.
pause
