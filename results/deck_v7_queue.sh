for s in MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE TDM; do
  .venv/bin/python -u scripts/deck_synergy_v7.py $s > results/deck_synergy_v7_$s.log 2>&1
done
