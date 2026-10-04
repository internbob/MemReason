import numpy as np
import pandas as pd

def split_subjects(subjects, seed=2026, fractions=(0.7,0.1,0.2)):
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError('Expected three positive split fractions summing to one')
    ids = np.array(sorted(set(subjects)), dtype=object)
    np.random.default_rng(seed).shuffle(ids)
    n_train, n_val = int(len(ids)*fractions[0]), int(len(ids)*fractions[1])
    labels = ['train']*n_train + ['val']*n_val + ['test']*(len(ids)-n_train-n_val)
    return pd.DataFrame({'subject_id':ids, 'split':labels}).sort_values('subject_id').reset_index(drop=True)
