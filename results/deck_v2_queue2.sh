while pgrep -f deck_v2_queue.sh > /dev/null; do sleep 30; done
for s in MSH BLB DFT DSK FDN FIN LCI MKM OTJ EOE SOS TDM; do
  .venv/bin/python -u scripts/deck_synergy_v2.py $s 0 > results/deck_synergy_v2_$s.rerun.log 2>&1
done
