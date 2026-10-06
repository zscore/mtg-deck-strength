# DraftEncoder multi-set replication queue (run under caffeinate)
.venv/bin/python -u -c "
import sys; sys.path.insert(0,'src')
from openlane.de_multiset import SetData
for s in ['WOE','LCI','MKM','OTJ','DSK','DFT','FIN','EOE']:
    d=SetData(s); print(s, d.packs.shape, flush=True)
" > results/de_multiset_cache.log 2>&1
.venv/bin/python -u scripts/train_draftencoder_multiset.py newset heuristic lr lr+meta bag bag+meta de de+meta per_set=20000 epochs=6 lin_epochs=10 ft_epochs=8 > results/de_multiset_newset.log 2>&1
# rerun the bag models with the running-mean parametrisation (bag+meta diverged with sums)
.venv/bin/python -u scripts/train_draftencoder_multiset.py newset bag bag+meta per_set=20000 lin_epochs=10 ft_epochs=8 > results/de_multiset_newset_bag.log 2>&1
