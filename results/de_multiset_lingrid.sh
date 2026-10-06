# linear baselines with a validation-chosen learning rate
while kill -0  2>/dev/null; do sleep 30; done
P=.venv/bin/python
$P -u scripts/train_draftencoder_multiset.py sameset bag bag+meta bag_id lin_grid=1e-3,3e-3,1e-2,3e-2 > results/de_multiset_sameset_lin.log 2>&1
$P -u scripts/train_draftencoder_multiset.py sameset lr lr+meta lin_grid=3e-5,1e-4,3e-4 > results/de_multiset_sameset_lr.log 2>&1
$P -u scripts/train_draftencoder_multiset.py newset bag bag+meta per_set=20000 lin_epochs=10 ft_epochs=8 lin_grid=1e-3,3e-3,1e-2,3e-2 > results/de_multiset_newset_lin.log 2>&1
$P -u scripts/train_draftencoder_multiset.py newset lr lr+meta per_set=20000 lin_epochs=10 ft_epochs=8 lin_grid=3e-5,1e-4,3e-4 > results/de_multiset_newset_lr.log 2>&1
