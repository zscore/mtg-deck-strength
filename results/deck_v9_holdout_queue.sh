for s in DFT BLB LCI; do
  .venv/bin/python -u scripts/deck_synergy_v9.py holdout=$s > results/deck_synergy_v9_holdout_$s.log 2>&1
done
