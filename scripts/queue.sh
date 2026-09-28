#!/usr/bin/env bash
# The only script that trains. It runs (method, seed) pairs of one config, one at a time:
#   bash scripts/queue.sh configs/<exp>.yaml [method:seed ...]
# Without pairs it runs every method of the config at each seed in turn (all methods at the
# first seed before any method at the next). Pairs that already have a results row are
# skipped, so a queue can be relaunched at any time.
#
# Several queues can run at once (lanes). Before starting a pair a queue takes a lock in
# logs/locks/ (one per output folder, method and seed); a pair locked by another live queue is
# skipped, so two lanes never train the same run. A lock left by a queue that died is taken over.
#
# Shared GPU: before each run it waits until the config's gpu_memory_cap_gb plus 1 GB is free
# on two checks a minute apart. A run that ends without a results row (out of memory or any
# other crash) is retried, up to 3 attempts; if the checkpoint was saved, only the evaluation
# reruns.
#
# The first run of a config trains its reference if it is missing. Start a second queue on the
# same config only after the reference and its log-prob cache exist.
#
# Logs: logs/<experiment_name>.log (training output), logs/<experiment_name>_queue.log (events).
# DRY_RUN=1 prints the order, what is done or running elsewhere and the memory it waits for.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
CFG=${1:?usage: bash scripts/queue.sh configs/<exp>.yaml [method:seed ...]}
shift

# experiment name, output folder, memory to wait for, pairs the config does not have, default order
read -r NAME OUT NEED_MIB BAD DEFAULT_PAIRS < <($PY - "$CFG" "$@" <<'PYEOF'
import sys
from pathlib import Path
from utils import load_experiment_config
cfg = load_experiment_config(sys.argv[1])
methods = [m for m in cfg["methods"] if m != "reference"]

def known(pair):
    m, _, s = pair.partition(":")
    return m in methods and s.isdigit() and int(s) in cfg["seeds"]

bad = [p for p in sys.argv[2:] if not known(p)]
print(cfg["experiment_name"], Path(cfg["output_dir"]).name, int((cfg.get("gpu_memory_cap_gb", 24) + 1) * 1024),
      ",".join(bad) or "-", " ".join(f"{m}:{s}" for s in cfg["seeds"] for m in methods))
PYEOF
)
[ -n "${NAME:-}" ] || { echo "could not read $CFG"; exit 1; }
[ "$BAD" = "-" ] || { echo "not a method:seed of $CFG: $BAD"; exit 1; }
PAIRS=${*:-$DEFAULT_PAIRS}

mkdir -p logs/locks
LOG=logs/${NAME}_queue.log
log() { echo "[$NAME $(date -Is)] $*" | tee -a "$LOG"; }

free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1 - $2}'; }

wait_memory() {
    while true; do
        a=$(free_mib); sleep 60; b=$(free_mib)
        [ "$a" -ge "$NEED_MIB" ] && [ "$b" -ge "$NEED_MIB" ] && return
        log "waiting for GPU memory: free $a / $b MiB, need $NEED_MIB"
        sleep 240
    done
}

has_result() {  # method seed
    $PY - "$CFG" "$1" "$2" <<'PYEOF'
import sys
from utils import load_experiment_config, result_exists
cfg = load_experiment_config(sys.argv[1])
m, s = sys.argv[2], int(sys.argv[3])
sys.exit(0 if result_exists(cfg["output_dir"] + "/results.jsonl", m, cfg["methods"][m], s) else 1)
PYEOF
}

lock_dir() { echo "logs/locks/${OUT}__$1__$2"; }
held_elsewhere() {  # method seed: 0 if a live queue other than this one holds the lock
    local pid; pid=$(cat "$(lock_dir "$1" "$2")/pid" 2>/dev/null)
    [ -n "$pid" ] && [ "$pid" != "$$" ] && kill -0 "$pid" 2>/dev/null
}
claim() {  # method seed: 0 if this queue now holds the lock
    local d; d=$(lock_dir "$1" "$2")
    if mkdir "$d" 2>/dev/null || ! held_elsewhere "$1" "$2"; then echo $$ > "$d/pid"; return 0; fi
    return 1
}
CURRENT=""
release() { [ -n "$CURRENT" ] && rm -rf "$(lock_dir ${CURRENT%:*} ${CURRENT#*:})"; CURRENT=""; }
trap 'release' EXIT
trap 'release; exit 130' INT TERM HUP

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "$NAME ($CFG): waits for $NEED_MIB MiB free before each run"
    for p in $PAIRS; do
        if has_result "${p%:*}" "${p#*:}"; then echo "  done     $p"
        elif held_elsewhere "${p%:*}" "${p#*:}"; then echo "  running  $p (another queue)"
        else echo "  to run   $p"; fi
    done
    exit 0
fi

log "queue start ($CFG): $PAIRS"
for p in $PAIRS; do
    m=${p%:*}; s=${p#*:}
    if has_result "$m" "$s"; then continue; fi
    if ! claim "$m" "$s"; then log "skip $m seed $s: another queue is running it"; continue; fi
    CURRENT=$p
    for attempt in 1 2 3; do
        if has_result "$m" "$s"; then break; fi
        wait_memory
        log "start $m seed $s (attempt $attempt)"
        $PY -u src/experiment_multipref.py run --config "$CFG" --method "$m" --seeds "$s" >> "logs/$NAME.log" 2>&1
        if has_result "$m" "$s"; then log "done $m seed $s"; else log "no result for $m seed $s after attempt $attempt"; fi
    done
    release
done
log "queue complete"
