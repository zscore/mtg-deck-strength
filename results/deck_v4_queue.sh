for s in MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE TDM; do
  .venv/bin/python -u scripts/deck_synergy_v4.py $s > results/deck_synergy_v4_$s.log 2>&1
done
.venv/bin/python -u scripts/deck_synergy_v4_draft.py MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE SOS TDM > results/deck_synergy_v4_draft.log 2>&1
