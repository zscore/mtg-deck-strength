# run the newset DraftEncoders after the sameset ones (both at once swap a 16GB machine)
while kill -0 19956 2>/dev/null; do sleep 30; done
.venv/bin/python -u scripts/train_draftencoder_multiset.py newset de de+meta per_set=20000 epochs=6 ft_epochs=8 lr=0.0003 > results/de_multiset_newset_de.log 2>&1
