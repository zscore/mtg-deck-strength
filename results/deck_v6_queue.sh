for s in SOS MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE TDM; do
  .venv/bin/python -u scripts/deck_synergy_v6.py $s > results/deck_synergy_v6_$s.log 2>&1
done
