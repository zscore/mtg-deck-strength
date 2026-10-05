for s in MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE SOS TDM; do
  .venv/bin/python -u scripts/deck_synergy_v2.py $s 0 > results/deck_synergy_v2_$s.rerun.log 2>&1
done
