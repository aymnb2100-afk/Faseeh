@echo off
cd /d "%~dp0"
echo  What the server is doing (close this window to stop watching; the server keeps running):
docker compose --env-file settings.env logs -f --tail 80
