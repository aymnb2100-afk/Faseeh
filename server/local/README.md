# Faseeh server on your own computer (NVIDIA graphics card)

The same analysis server as the Hugging Face Space, on this computer's graphics card: seconds instead of
minutes. The website uses it by itself while it is on, and goes back to the Space when it is off.

**Isolated:** it runs in Docker containers. It cannot see your files, and no port of this computer is opened;
only ngrok gives the server (and nothing else) a fixed public address.

**Always up to date:** the container takes the server code, settings and calibration from GitHub
(`server/hf_space`) and checks every 2 minutes; a change is applied by itself (the server restarts in about
half a minute). Nothing has to be downloaded again here.

## Once
1. Update the NVIDIA driver; in PowerShell as administrator: `wsl --install`, restart.
2. Install Docker Desktop (WSL 2).
3. Put your ngrok Authtoken and fixed domain in `settings.env` (keep it private).
4. Double-click `start_faseeh.bat`.

## Files
| file | does |
|---|---|
| `start_faseeh.bat` | builds and starts everything, shows when it is ready |
| `stop_faseeh.bat` | stops it (the website goes back to the Space) |
| `logs_faseeh.bat` | shows what the server is doing |
| `remove_faseeh.bat` | deletes everything: containers, model, image |
| `settings.env` | your ngrok details (never upload it) |
| `Dockerfile`, `boot.sh`, `docker-compose.yml` | the container: tools only, no server code |
| `supervisor.py` | runs from GitHub inside the container: updates and restarts the server |

It never starts by itself: it runs only after `start_faseeh.bat` and until `stop_faseeh.bat` (or until the computer
is turned off). Leave "Start Docker Desktop when you sign in" off in Docker Desktop's settings.
