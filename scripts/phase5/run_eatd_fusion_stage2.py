import pickle, sys
from pathlib import Path
sys.path.insert(0, ".")
state = pickle.load(open("results/phase5/eatd_fusion/state.pkl", "rb"))
globals().update(state)
src = Path("scripts/phase5/run_eatd_fusion.py").read_text().splitlines()
# compute reps needed for fusion loop (cheap, no model training)
from ml.evaluation.splits import repeated_stratified_group_folds, eatd_locked_test_group_ids
locked = eatd_locked_test_group_ids()
OUTER_FOLDS = 5
OUTER_SEEDS = (42, 43, 44, 45, 46)
reps_t1 = repeated_stratified_group_folds(units_t1, n_splits=OUTER_FOLDS, seeds=OUTER_SEEDS, locked_group_ids=locked)
reps_v = repeated_stratified_group_folds(units_v, n_splits=OUTER_FOLDS, seeds=OUTER_SEEDS, locked_group_ids=locked)
# E-T2 reps not needed for fusion loop? It uses reps_t1 for outer loop; inner uses inner_part_t2 already. Good.
top = "\n".join(src[:163])
post = "\n".join(src[463:])
exec(compile(top + "\n" + post, "stage2", "exec"), globals())
