#!/bin/bash
# Full pipeline on the pod: prep -> embeddings (xsmE) -> blocker (mini, minid, train, test) -> eval sets.
# Logs to $SABER_WORK/pod.log; every step is skipped if its output exists, so the script can be re-run after a stop.
set -uo pipefail
cd /workspace/saber && source pod/env.sh
W=$SABER_WORK; M=$W/ft_arctic-xs-mean_E3k
log() { echo "[$(date '+%F %T')] $*" | tee -a $W/pod.log; }
run() { log "$*"; "$@" >> $W/pod_steps.log 2>&1 || { log "FAIL: $*"; exit 1; }; }
[ -f $W/test_source3.parquet ] || run python src/prep.py
python - <<'PY' | tee -a $W/pod.log                                      # fingerprints, compared with the laptop's
import hashlib, os, polars as pl
W = os.environ["SABER_WORK"] + "/"
for sp in ("train", "test"):
    for s in ("source1", "source2", "source3"):
        d = pl.read_parquet(W + f"{sp}_{s}.parquet", columns=["entity_id", "name", "addr"])
        h = hashlib.sha1()
        for c in d.columns: h.update("
".join(d[c].to_list()).encode())
        print(sp, s, d.height, h.hexdigest()[:16])
PY
for sp in mini train test; do
  [ -f $W/emb_xsmE_${sp}_source3.npy ] || run python src/encode.py --model $M --tag xsmE --split $sp
done
[ -f $W/emb_xsmE_minid_source1.npy ] || run python src/make_split.py minid --base mini --drop-s1 0.19 --emb xsmE
for sp in mini minid train test; do
  [ -f $W/cand_xsmE_${sp}_manifest.json ] && { log "have $sp candidates"; continue; }
  run python src/blocker.py search --split $sp --emb xsmE
  run python src/blocker.py merge --split $sp --emb xsmE
  log "$sp ready: $(python -c "import json;r=json.load(open('$W/block_report_xsmE_$sp.json'));print({c:(v['recall_all'],v['per_s1']) for c,v in r.items()})" 2>/dev/null)"
done
run python src/eval_sets.py --emb xsmE --split mini minid train
log "POD_V2_READY"
