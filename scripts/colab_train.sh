#!/usr/bin/env bash
# Run the training pipeline on a Colab GPU via the `colab` CLI, fetch the results, release the VM.
#
#   scripts/colab_train.sh smoke
#   scripts/colab_train.sh full --push-to-hub zackyalgiffari/ott-recommender-ml32m
#   scripts/colab_train.sh ablations
#
# The first argument is the mode; the rest is forwarded to scripts/remote_job.sh.
# Everything the job writes under results/ (plus run.log) is downloaded to runs/<session>/.
# Env: GPU (default L4), SESSION (default ott-<mode>-<time>), POLL_SECONDS (60),
#      CALL_TIMEOUT (120), LAUNCH_TIMEOUT (300), HF_TOKEN_FILE.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: colab_train.sh smoke|full|ablations [args...]}"
GPU="${GPU:-L4}"
SESSION="${SESSION:-ott-$MODE-$(date +%H%M%S)}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CALL_TIMEOUT="${CALL_TIMEOUT:-120}"     # max seconds for one poll/download call
LAUNCH_TIMEOUT="${LAUNCH_TIMEOUT:-300}" # max seconds for the launch call (extract + spawn)
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
DEST="runs/$SESSION"
JOB_ARGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")"

mkdir -p dist "$DEST"
# Pack tracked + untracked-but-not-ignored files (works before the first commit too).
git ls-files -z -co --exclude-standard | tar --null -T - -czf dist/src.tar.gz

echo "[local] creating session $SESSION ($GPU)"
colab new -s "$SESSION" --gpu "$GPU"
trap 'echo "[local] stopping $SESSION"; colab stop -s "$SESSION" || true' EXIT

colab upload -s "$SESSION" dist/src.tar.gz /content/src.tar.gz
if [[ -f "$HF_TOKEN_FILE" ]]; then
  colab upload -s "$SESSION" "$HF_TOKEN_FILE" /content/.hf_token
fi

# `colab exec` can hang after the remote cell finishes, or lose its websocket before the
# cell runs at all, so every call is bounded by `timeout` and the launch is retried until
# /content/run.log exists. The cell refuses to start a second job if one already started.
LAUNCH_PY="$(cat <<'PY'
import json, os, subprocess
def launch():
    if os.path.exists("/content/run.log"):
        print("ALREADY LAUNCHED")
        return
    env = dict(os.environ, HF_TOKEN_PATH="/content/.hf_token", PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1",
               OTT_DATA_DIR="/content/data")
    if os.path.exists("/content/.hf_token"):
        os.chmod("/content/.hf_token", 0o600)
    subprocess.run(["bash", "-c", "rm -rf /content/app && mkdir -p /content/app && "
                    "tar -xzf /content/src.tar.gz -C /content/app"], check=True)
    args = json.loads(os.environ["JOB_ARGS"])
    with open("/content/run.log", "w") as log:
        subprocess.Popen(
            ["bash", "-c", 'bash scripts/remote_job.sh "$@"; echo $? > /content/run.exit', "job", *args],
            cwd="/content/app", env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    print("LAUNCHED", args)
launch()
PY
)"
CHECK_PY='import os; print("HAS_LOG" if os.path.exists("/content/run.log") else "NO_LOG")'
for attempt in 1 2 3; do
  printf '%s\n' "$LAUNCH_PY" | timeout "$LAUNCH_TIMEOUT" colab exec -s "$SESSION" --env "JOB_ARGS=$JOB_ARGS_JSON" \
    || echo "[local] launch attempt $attempt did not return cleanly"
  if printf '%s' "$CHECK_PY" | timeout "$CALL_TIMEOUT" colab exec -s "$SESSION" 2>/dev/null | grep -q HAS_LOG; then
    echo "[local] job started (attempt $attempt)"; break
  fi
  echo "[local] job not started yet; retrying launch"; sleep 20
done

POLL_PY='
import os
lines = open("/content/run.log", errors="replace").read().splitlines() if os.path.exists("/content/run.log") else []
print("\n".join(l[-220:] for l in lines[-4:] if l.strip()))
if os.path.exists("/content/run.exit"):
    print("EXIT=" + open("/content/run.exit").read().strip())
else:
    print("RUNNING" if os.path.exists("/content/run.log") else "NOLOG")
'
nolog=0
while true; do
  sleep "$POLL_SECONDS"
  out="$(printf '%s' "$POLL_PY" | timeout "$CALL_TIMEOUT" colab exec -s "$SESSION" 2>&1 || true)"
  echo "----- $(date +%T)"; echo "$out" | grep -vE '^\s*$' | tail -6
  if [[ "$out" == *NOLOG* ]] && (( ++nolog >= 5 )); then
    echo "[local] job never started (no /content/run.log)"; exit 1
  fi
  if [[ "$out" == *"EXIT="[0-9]* ]]; then
    code="$(grep -o 'EXIT=[0-9]*' <<<"$out" | tail -1 | cut -d= -f2)"
    break
  fi
done

timeout "$CALL_TIMEOUT" colab download -s "$SESSION" /content/run.log "$DEST/run.log" || true
LIST_PY='
import os
root = "/content/app/results"
for d, _, files in os.walk(root):
    for f in files:
        print("FILE=" + os.path.relpath(os.path.join(d, f), root))
'
files="$(printf '%s' "$LIST_PY" | timeout "$CALL_TIMEOUT" colab exec -s "$SESSION" 2>/dev/null | sed -n 's/^FILE=//p' || true)"
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  mkdir -p "$DEST/$(dirname "$f")"
  timeout "$CALL_TIMEOUT" colab download -s "$SESSION" "/content/app/results/$f" "$DEST/$f" \
    || echo "[local] could not download $f"
done <<<"$files"
echo "[local] job exit code $code; artifacts in $DEST/"
exit "$code"
