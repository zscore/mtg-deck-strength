for s in SOS MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE TDM; do
  .venv/bin/python -u scripts/deck_synergy_v8.py $s 2>&1 | grep -v -E "RuntimeWarning|c /= stddev" > results/deck_synergy_v8_$s.log
done
