for s in MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE TDM; do
  .venv/bin/python -u scripts/deck_synergy_v3.py $s > results/deck_synergy_v3_$s.log 2>&1
done
.venv/bin/python -u scripts/deck_synergy_v3_blend.py MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE SOS TDM > results/deck_synergy_v3_blend.log 2>&1
