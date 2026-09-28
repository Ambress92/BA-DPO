#!/usr/bin/env bash
# Links checkpoints of a MultiPref length experiment into its formatting experiment, so the
# formatting config evaluates the SAME checkpoints under another attribute instead of retraining
# them: dpo, R-DPO and SamPO never read the declared attribute (see the comment in
# configs/multipref_formatting.yaml). Arms are linked as whole folders, so seeds that finish
# later appear on their own. Safe to run again: existing links are kept.
# Usage: bash scripts/setup/link_multipref_formatting.sh [length_experiment formatting_experiment]
#   0.5B (default): multipref multipref_formatting
#   8B:             multipref_llama8b multipref_formatting_llama8b
set -eu
cd "$(dirname "$0")/../.."
SRC=${1:-multipref}
DST=${2:-multipref_formatting}

mkdir -p "experiments/$DST"
for arm in dpo rdpo rdpo_a005 sampo; do
    if [ -d "experiments/$SRC/$arm" ] && [ ! -e "experiments/$DST/$arm" ]; then
        ln -s "$(pwd)/experiments/$SRC/$arm" "experiments/$DST/$arm"
        echo "linked experiments/$DST/$arm -> experiments/$SRC/$arm"
    fi
done

# 0.5B only: that formatting config names its reference folder after its data build, so the
# folder is linked to the length reference. The 8B configs set reference_name instead.
if [ "$DST" = multipref_formatting ] && [ ! -e experiments/shared/multipref_formatting ]; then
    ln -s "$(pwd)/experiments/shared/multipref" experiments/shared/multipref_formatting
    echo "linked experiments/shared/multipref_formatting -> experiments/shared/multipref"
fi
