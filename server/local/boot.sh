#!/bin/sh
# Gets the repository (only its server folder) and runs its supervisor; runs the new one whenever it updates itself.
R="${FASEEH_SRC:-/src/repo}"
while true; do
  if [ ! -d "$R/.git" ]; then
    rm -rf "$R"
    if git clone --depth 1 --filter=blob:none --sparse --branch "$FASEEH_BRANCH" "$FASEEH_REPO" "$R" \
       && git -C "$R" sparse-checkout set server; then
      echo "got the server from GitHub"
    else
      echo "GitHub not reachable, trying again in 30 s"; rm -rf "$R"; sleep 30; continue
    fi
  else
    git -C "$R" fetch --depth 1 origin "$FASEEH_BRANCH" && git -C "$R" reset --hard FETCH_HEAD \
      || echo "GitHub not reachable: using the copy already here"
  fi
  python "$R/server/local/supervisor.py"
  echo "supervisor stopped (code $?), starting it again"
  sleep 3
done
