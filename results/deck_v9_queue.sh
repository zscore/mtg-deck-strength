.venv/bin/python -u scripts/deck_synergy_v9.py > results/deck_synergy_v9.log 2>&1
for s in BLB DFT DSK EOE FDN FIN LCI MKM MSH OTJ SOS TDM; do
  .venv/bin/python -u scripts/deck_synergy_v9.py holdout=$s > results/deck_synergy_v9_holdout_$s.log 2>&1
done
