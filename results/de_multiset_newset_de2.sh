# newset DraftEncoders after the sameset ones (both at once swap a 16GB machine); smaller to fit the night
while kill -0 19956 2>/dev/null; do sleep 30; done
.venv/bin/python -u scripts/train_draftencoder_multiset.py newset de de+meta per_set=12000 epochs=6 ft_epochs=8 scratch_epochs=10 lr=0.0003 > results/de_multiset_newset_de.log 2>&1
# linear baselines on the same 12k-per-set sample as the DraftEncoders, for a like-for-like row
.venv/bin/python -u scripts/train_draftencoder_multiset.py newset bag bag+meta per_set=12000 lin_epochs=10 ft_epochs=8 scratch_epochs=10 lin_grid=1e-3,3e-3,1e-2,3e-2 tag=_12k > results/de_multiset_newset_lin12k.log 2>&1
# the paper's leaky META (whole game file) for the linear models, to size the leak
.venv/bin/python -u scripts/train_draftencoder_multiset.py newset heuristic lr+meta bag+meta per_set=12000 lin_epochs=10 ft_epochs=8 scratch_epochs=10 lin_grid=1e-3,3e-3,1e-2,3e-2 meta_all=1 tag=_leakymeta > results/de_multiset_newset_leaky.log 2>&1
