"""Frozen graph of the externally best joint_candidate_forecast submission."""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from .all40_model import NativePart
from .research_model import WeeklyResearchModel, recipe_prediction
from ..daily_bridge import aggregate_daily
from ..joint_features import JOINT_DAILY_COLUMNS, make_joint_daily


def joint_graph():
    weights = [[.75, .75, .5] for _ in range(5)]
    lambdas = [[3., 3., 3.] for _ in range(5)]
    lambdas[1][2] = lambdas[2][2] = 6.
    return dict(weekly_policy=dict(weights=weights, lambdas=lambdas, threshold=.65),
                daily_variant='weekly_poisson_aux', weekly_event_gate=.5,
                daily_weekly_rounding=True, daily_weight=.35, final_rounding=True)


def decode(probability, schema, multiplier):
    representatives = np.asarray(schema['representatives'], float)
    actions = np.unique(np.r_[0., .5, 1., 2., 3., 4., 6., 8., 12., 16., 24., 32., 48., representatives])
    error = np.abs(representatives[:, None] - actions[None, :])
    den = representatives[:, None] + actions[None, :]
    relative = np.divide(200 * error, den, out=np.zeros_like(error), where=den > 0)
    risk = .7 / 60 * relative + multiplier * .3 / 50 * 100 * error / max(schema['training_mean'], 1e-9)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api='blas'):
        return actions[np.argmin(probability @ risk, axis=1)]


def encoded(frame, schema):
    x = frame[schema['columns']].copy()
    for column, categories in schema['categories'].items():
        x[column] = pd.Categorical(x[column].astype(str), categories=categories)
    return x


class DailyQuantityPart(NativePart):
    def check_schema(self):
        if self.schema['columns'] != JOINT_DAILY_COLUMNS or self.booster.feature_name() != JOINT_DAILY_COLUMNS:
            raise ValueError('Daily expert feature schema differs')
        if self.schema.get('prediction') != 'exp(clip(raw_booster_score + past_only_offset, -15, 15))':
            raise ValueError('Daily expert output transform differs')
        if self.schema.get('variant', {}).get('id') != 'weekly_poisson_aux':
            raise ValueError('Unexpected daily quantity expert')

    def predict(self, frame):
        raw = self.booster.predict(encoded(frame, self.schema), raw_score=True, num_threads=self.threads)
        offset = np.log(np.maximum(.5 * frame.mean_28.to_numpy(float) + .5 * frame.dow_mean.to_numpy(float), .02))
        return np.exp(np.clip(raw + offset, -15, 15))


class JointWeeklyDailyModel(WeeklyResearchModel):
    def __init__(self, weekly, daily):
        params = copy.deepcopy(weekly.params)
        params['serving_profile'] = 'joint_v1'
        params['joint_graph'] = joint_graph()
        super().__init__(params)
        self.weekly = weekly
        self.daily = daily

    def fit(self, *args, **kwargs):
        raise RuntimeError('Import verified user-trained weights; serving never trains.')

    def check_schema(self):
        self.weekly.check_schema()
        self.daily.check_schema()
        expected = copy.deepcopy(self.weekly.params)
        expected.update(serving_profile='joint_v1', joint_graph=joint_graph())
        if self.params != expected or self.recipe != self.weekly.recipe:
            raise ValueError('Joint model graph differs from the frozen submission')

    def weekly_prediction(self, features):
        # All40 training used float32 numerical features, as did the candidate builder.
        frame = features.copy()
        for column in frame.select_dtypes(include=[np.number]):
            frame[column] = frame[column].astype('float32')
        parts = self.weekly.parts
        legacy = self.weekly.baseline.predict(features)
        r = 7 * features.mean_28.to_numpy(float)
        scope = (r >= .5) & (r < 10)
        tail = features.h.to_numpy(float) >= 7
        count = parts['count_fine']
        probability = count.booster.predict(encoded(frame, count.schema), num_threads=count.threads)
        left = legacy.copy()
        for multiplier, mask in ((3., scope & ~tail), (6., scope & tail)):
            if mask.any():
                left[mask] = decode(probability[mask], count.schema, multiplier)
        p = {name: part.predict(frame) for name, part in parts.items() if name != 'count_fine'}
        values = {'dynamics': dict(probability=p['core_event'], magnitude=p['core_positive'], lgbm_raw=p['core_log']),
                  'mae': dict(raw=p['core_mae']), 'metric_count_lambda3': dict(raw=p['core_count']),
                  'metric_regular_recent_mae': dict(raw=p['core_recent'])}
        refreshed = recipe_prediction(values, frame, self.recipe)
        weight = np.where(tail, .5, .75)
        # Preserve original candidate's operation order and threshold (not bankers rounding).
        return np.floor(np.maximum(weight * left + (1 - weight) * refreshed, 0.) + 1 - .65)

    def predict(self, features):
        raise ValueError('Joint inference requires raw request history; use Predictor.predict.')

    def predict_from_history(self, features, sales, cutoff):
        weekly = self.weekly_prediction(features)
        daily = make_joint_daily(sales, features, cutoff)
        summed = aggregate_daily(daily, self.daily.predict(daily), features)
        event = self.weekly.parts['core_event'].predict(features)
        daily_weekly = np.rint(np.where(event >= .5, summed, 0.))
        return np.rint(.65 * weekly + .35 * daily_weekly)

    @property
    def selected_iterations(self):
        return {**self.weekly.selected_iterations, 'daily_weekly_poisson_aux': self.daily.booster.current_iteration()}
