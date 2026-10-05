while [ ! -f results/picknet_logs/pool_models.done ]; do sleep 30; done
for s in BLB DFT DSK FDN FIN LCI MKM OTJ EOE SOS TDM; do
  .venv/bin/python -u scripts/deck_synergy_v2.py $s > results/deck_synergy_v2_$s.log 2>&1
done
