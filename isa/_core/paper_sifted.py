import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


REFERENCE = 'https://github.com/andremun/InstanceSpace/blob/e7fa8002d8979576ce53115f745ce6f9f3331df0/SIFTED.m'
PAPER_REFERENCE = 'https://doi.org/10.1016/j.ejor.2023.12.008'
IMPLEMENTATION_VERSION = '20260916_small_set_and_tree_defaults'


def correlation_clusters(x, count, seed):
    profiles = x.T.copy()
    profiles -= profiles.mean(axis=1, keepdims=True)
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True)
    count = min(count, len(np.unique(np.round(profiles, 12), axis=0)))
    if count < 2:
        raise ValueError('Surviving features do not support two distinct correlation clusters')
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(10):
        centers = profiles[rng.choice(len(profiles), count, replace=False)].copy()
        previous = None
        for iteration in range(100):
            similarity = profiles @ centers.T
            labels = similarity.argmax(axis=1)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
            for cluster in range(count):
                members = profiles[labels == cluster]
                if len(members):
                    center = members.mean(axis=0)
                    norm = np.linalg.norm(center)
                    centers[cluster] = center / norm if norm else members[0]
                else:
                    centers[cluster] = profiles[np.argmin(similarity.max(axis=1))]
        labels = (profiles @ centers.T).argmax(axis=1)
        loss = float(np.sum(1 - (profiles @ centers.T).max(axis=1)))
        if len(set(labels)) == count and (best is None or loss < best[0]):
            best = (loss, labels.copy())
    if best is None:
        raise ValueError('Correlation clustering did not produce the requested nonempty clusters')
    return [np.flatnonzero(best[1] == c).tolist() for c in range(count)]


def select_features(features, good, seed=0):
    if not features.index.equals(good.index) or not features.index.is_unique:
        raise ValueError('SIFTED requires identical unique instance IDs in X and binary Y')
    if not features.columns.is_unique or not good.columns.is_unique or good.shape[1] == 0:
        raise ValueError('SIFTED requires unique feature/model names and at least one measured model')
    x, y = features.to_numpy(float), good.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('SIFTED requires finite features and measured 0/1 performance labels')
    active = np.flatnonzero(np.ptp(y, axis=0) > 0)
    if not len(active):
        raise ValueError('All formulation good/bad labels are constant; no binary performance trend for SIFTED')
    correlations = pd.DataFrame(0.0, index=features.columns, columns=good.columns)
    pvalues = pd.DataFrame(1.0, index=features.columns, columns=good.columns)
    for j in active:
        for i, name in enumerate(features.columns):
            if np.ptp(x[:, i]) == 0:
                continue
            statistic, pvalue = pearsonr(x[:, i], y[:, j])
            correlations.iloc[i, j] = abs(float(statistic))
            pvalues.iloc[i, j] = float(pvalue)
    survivors = [f for f in features if correlations.loc[f].max() >= 0.2]
    if len(survivors) < 2:
        raise ValueError(f'Only {len(survivors)} features pass the fixed 0.2 correlation threshold; no fallback')
    metadata = {'surviving_features': survivors, 'correlations': correlations.to_dict(), 'threshold': 0.2,
                'pvalues_diagnostic_only': pvalues.to_dict(), 'pvalue_cutoff': None,
                'screening_response': 'measured binary good/bad labels, 2BP paper Section 4.1',
                'constant_label_models': [good.columns[j] for j in range(y.shape[1]) if j not in active],
                'reference': REFERENCE, 'paper_reference': PAPER_REFERENCE,
                'implementation_version': IMPLEMENTATION_VERSION,
                'tree_predictors_per_split': 2,
                'implementation_note': '2BP paper binary-response threshold and maximum 10 clusters; historical SIFTED PCA/OOB search. '
                'Below 10 surviving features, retain all as in the referenced author implementation; at 10, enter clustering. '
                'Python spherical k-means, sklearn 50-tree forest and custom bounded GA are not MATLAB-identical. '
                'Tree split predictor count uses ceil(sqrt(2))=2 to match TreeBagger defaults. '
                'The generic continuous-response p-value gate and always-keep-best fallback remain disabled to obey Section 4.1.',
                'circle_used_for_selection': False, 'performance_partition_generated': False}
    if len(survivors) < 10:
        return survivors, {**metadata, 'selected_features': survivors, 'clusters': [],
                           'representative_search': 'not_required_fewer_than_10_survivors',
                           'candidate_subsets_evaluated': 0, 'oob_error_sum': None}
    sub = features[survivors].to_numpy(float)
    clusters = correlation_clusters(sub, min(10, len(survivors)), seed)
    sizes = [len(c) for c in clusters]
    cache = {}

    def score(choice):
        choice = tuple(choice)
        if choice in cache:
            return cache[choice]
        indices = [members[v] for members, v in zip(clusters, choice)]
        projected = PCA(n_components=2, svd_solver='full').fit_transform(sub[:, indices])
        error = 0.0
        for j in active:
            forest = RandomForestClassifier(n_estimators=50, oob_score=True, bootstrap=True,
                                            max_features=math.ceil(math.sqrt(projected.shape[1])), random_state=seed, n_jobs=1)
            forest.fit(projected, y[:, j].astype(int))
            if np.any(forest.oob_decision_function_.sum(axis=1) == 0):
                raise ValueError('Missing OOB predictions in SIFTED representative evaluation')
            error += 1 - float(forest.oob_score_)
        cache[choice] = error
        return error

    combinations = math.prod(sizes)
    if combinations <= 1000:
        choices = itertools.product(*(range(s) for s in sizes))
        best = min(choices, key=lambda c: (score(c), c))
        search = 'exhaustive'
    else:
        rng = np.random.default_rng(seed)
        population = [tuple(rng.integers(s) for s in sizes) for _ in range(50)]
        best, best_value, stalls = None, float('inf'), 0
        for generation in range(100):
            ranked = sorted(set(population), key=lambda c: (score(c), c))
            if score(ranked[0]) < best_value - 1e-12:
                best, best_value, stalls = ranked[0], score(ranked[0]), 0
            else:
                stalls += 1
            if stalls >= 5:
                break
            elite = ranked[:max(2, min(10, len(ranked)))]
            population = elite[:2]
            while len(population) < 50:
                a, b = elite[rng.integers(len(elite))], elite[rng.integers(len(elite))]
                child = [a[k] if rng.random() < 0.5 else b[k] for k in range(len(sizes))]
                for k, s in enumerate(sizes):
                    if rng.random() < 1 / len(sizes):
                        child[k] = int(rng.integers(s))
                population.append(tuple(child))
        search = 'bounded_genetic_search'
    selected = [survivors[c[v]] for c, v in zip(clusters, best)]
    selected = [f for f in features if f in selected]
    return selected, {**metadata, 'selected_features': selected,
                      'clusters': [[survivors[i] for i in c] for c in clusters],
                      'representative_search': search, 'candidate_subsets_evaluated': len(cache),
                      'oob_error_sum': score(best)}
