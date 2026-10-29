"""Serializable past-only count and recent-L1 experts for research recipes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import CAT_FEATURES
from ..event_features import FEATURE_COLUMNS

EDGES = np.array([1., 2., 3., 4., 7., 15.])
FALLBACK = np.array([0., 1., 2., 3., 4., 6., 12., 24.])
ACTIONS = np.array([0., 1., 2., 3., 4., 6., 8., 12., 16., 24., 32., 48.])
COMMON = dict(metric='None', num_leaves=31, max_depth=7, reg_alpha=5., reg_lambda=20.,
              learning_rate=.05, min_child_samples=100, subsample=.9, subsample_freq=1,
              colsample_bytree=.8, seed=42, deterministic=True, force_col_wise=True, verbosity=-1)


def quantity_classes(y):
    y = np.asarray(y, float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid count target')
    return np.where(y == 0, 0, np.searchsorted(EDGES, y, side='left')+1).astype('int8')


def class_representatives(y):
    y = np.asarray(y, float)
    labels = quantity_classes(y)
    representatives = FALLBACK.copy()
    for label in range(1, 8):
        if (labels == label).any():
            representatives[label] = y[labels == label].mean()
    return representatives


def bayes_quantities(probabilities, representatives, train_mean, multiplier):
    probabilities = np.asarray(probabilities, float)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 8, order='F')
    if probabilities.ndim != 2 or probabilities.shape[1] != 8 or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('Invalid count probabilities')
    representatives = np.asarray(representatives, float)
    actions = np.unique(np.r_[ACTIONS, representatives])
    denominator = representatives[:, None]+actions[None, :]
    absolute = np.abs(representatives[:, None]-actions[None, :])
    smape = np.divide(200*absolute, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    risk = .7/60*smape + float(multiplier)*.3/50*100*absolute/max(float(train_mean), 1e-9)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api='blas'):
        return actions[np.argmin(probabilities@risk, axis=1)]


def point_score(y, prediction):
    y, prediction = np.asarray(y, float), np.maximum(np.asarray(prediction, float), 0.)
    d = y+prediction
    smape = np.divide(200*np.abs(y-prediction), d, out=np.zeros_like(y), where=d > 0).mean()
    wape = 100*np.abs(y-prediction).sum()/max(y.sum(), 1e-12)
    return float(.7*smape/60+.3*wape/50)


class MetricExpertPart:
    def __init__(self, spec, threads=6):
        self.spec = dict(spec)
        if self.spec.get('kind') not in ('count', 'recent_mae', 'recent_balanced_mae'):
            raise ValueError('Unknown metric expert')
        if self.spec.get('kind') == 'count' and (not np.isfinite(float(self.spec.get('multiplier', 1.))) or float(self.spec.get('multiplier', 1.)) < 0):
            raise ValueError('Invalid count decoder multiplier')
        self.threads = int(threads)
        self.params = {'feature_columns': list(FEATURE_COLUMNS)}
        self.categories = {}
        self.booster = None
        self.representatives = None
        self.training_mean = None

    def encode(self, frame, fit=False):
        result = frame[FEATURE_COLUMNS].copy()
        for column in CAT_FEATURES:
            if fit:
                self.categories[column] = sorted(result[column].astype(str).unique())
            result[column] = pd.Categorical(result[column].astype(str), categories=self.categories[column])
        return result

    def fit(self, train, valid=None, iterations=None):
        import lightgbm as lgb
        kind = self.spec['kind']
        if kind != 'count':
            train = train.loc[train.mean_28.mul(7).ge(2)].copy()
            valid = valid.loc[valid.mean_28.mul(7).ge(2)].copy() if valid is not None else None
        if train.empty:
            raise ValueError('Metric expert has no eligible training examples')
        y = train.target.to_numpy(float)
        self.training_mean = float(y.mean())
        x = self.encode(train, fit=True)
        weight = None
        if kind == 'count':
            self.representatives = class_representatives(y)
            labels = quantity_classes(y)
            parameters = {**COMMON, 'objective': 'multiclass', 'num_class': 8, 'num_threads': self.threads}
            maximum = 200
        else:
            labels = y
            boundary = (pd.to_datetime(train.week)+pd.Timedelta(days=6)).max()
            age = (boundary-pd.to_datetime(train.feature_origin)).dt.days.to_numpy(float)
            if (age < 0).any():
                raise ValueError('Metric expert training origins exceed label boundary')
            weight = 2.**(-age/180.)
            if kind == 'recent_balanced_mae':
                weight *= .7/(7*train.mean_28.to_numpy(float)+1)+.3/max(self.training_mean, 1e-9)
            weight /= weight.mean()
            parameters = {**COMMON, 'objective': 'regression_l1', 'num_threads': self.threads}
            maximum = 250
        dataset = lgb.Dataset(x, label=labels, weight=weight, categorical_feature=CAT_FEATURES, free_raw_data=True)
        options = {}
        callbacks = []
        if valid is not None and iterations is None:
            if valid.empty:
                raise ValueError('Metric expert has no eligible validation examples')
            xv = self.encode(valid)
            yy = valid.target.to_numpy(float)
            validation = lgb.Dataset(xv, label=quantity_classes(yy) if kind == 'count' else yy,
                                     reference=dataset, categorical_feature=CAT_FEATURES)
            state = {'iteration': 0, 'score': None}
            def evaluator(prediction, data):
                state['iteration'] += 1
                if kind != 'count' or state['iteration'] == 1 or state['iteration'] % 5 == 0:
                    decoded = bayes_quantities(prediction, self.representatives, self.training_mean, 1.) if kind == 'count' else prediction
                    state['score'] = point_score(yy, decoded)
                return 'smape_wape', state['score'], False
            options.update(valid_sets=[validation], feval=evaluator)
            callbacks.append(lgb.early_stopping(30, first_metric_only=True, verbose=False))
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api='blas'):
            self.booster = lgb.train(parameters, dataset, num_boost_round=int(iterations or maximum),
                                     callbacks=callbacks, **options)
        return self

    @property
    def selected_iterations(self):
        chosen = int(self.booster.best_iteration or 0)
        return chosen if chosen > 0 else int(self.booster.current_iteration())

    def predict(self, frame):
        raw = self.booster.predict(self.encode(frame), num_threads=self.threads)
        if self.spec['kind'] == 'count':
            return bayes_quantities(raw, self.representatives, self.training_mean, self.spec.get('multiplier', 1.))
        return np.maximum(np.asarray(raw, float), 0.)
