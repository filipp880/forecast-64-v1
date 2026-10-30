"""Self-contained inference for the verified all40 fixed-weight ensemble."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from .research_model import WeeklyResearchModel, recipe_prediction
from ..features import CAT_FEATURES
from ..advanced_features import ADVANCED_COLUMNS


def decode_count(probability,schema):
    representatives=np.asarray(schema['representatives'],float)
    actions=np.unique(np.r_[0.,.5,1.,2.,3.,4.,6.,8.,12.,16.,24.,32.,48.,representatives])
    absolute=np.abs(representatives[:,None]-actions[None,:]);den=representatives[:,None]+actions[None,:]
    smape=np.divide(200*absolute,den,out=np.zeros_like(absolute),where=den>0)
    risk=.7/60*smape+3.*.3/50*100*absolute/max(schema['training_mean'],1e-9)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):
        return actions[np.argmin(probability@risk,axis=1)]


class NativePart:
    def __init__(self,model_text,schema,threads=6):
        self.model_text=model_text;self.schema=copy.deepcopy(schema);self.threads=int(threads);self._booster=None

    @property
    def booster(self):
        if self._booster is None:
            import lightgbm as lgb
            self._booster=lgb.Booster(model_str=self.model_text)
        return self._booster

    def __getstate__(self):return {**self.__dict__,'_booster':None}

    def check_schema(self):
        if self.booster.feature_name()!=self.schema['columns']:raise ValueError('Native model feature schema differs')
        if not set(self.schema['columns'])<=set(ADVANCED_COLUMNS):raise ValueError('Unsupported native feature column')
        if self.schema['spec']['kind']=='count':
            if self.booster.num_model_per_iteration()!=len(self.schema['representatives']):raise ValueError('Native count classes differ')

    def predict(self,frame):
        x=frame[self.schema['columns']].copy()
        for column,categories in self.schema['categories'].items():
            x[column]=pd.Categorical(x[column].astype(str),categories=categories)
        p=self.booster.predict(x,num_threads=self.threads)
        spec=self.schema['spec']
        if spec['kind']=='count':return decode_count(p,self.schema)
        if spec.get('log_target'):return np.maximum(np.expm1(np.clip(p,-20,20)),0.)
        return p if spec['kind']=='binary' else np.maximum(p,0.)


class All40WeeklyModel(WeeklyResearchModel):
    """Research-compatible artifact: 75% refined-count branch + 25% refreshed recipe."""
    def __init__(self,baseline,parts):
        params=copy.deepcopy(baseline.params)
        params['serving_profile']='all40_v1'
        params['all40_blend']=dict(count_branch=.75,refreshed_branch=.25,count_min_mean=.5,
            count_max_mean=10.,round_int=True,count_decoder_multiplier=3.)
        super().__init__(params)
        self.baseline=baseline;self.parts=dict(parts)

    def fit(self,*args,**kwargs):
        raise RuntimeError('Import the verified user-trained all40 components; do not refit through the serving class.')

    def check_schema(self):
        self.baseline.check_schema()
        expected={'count_fine','core_event','core_positive','core_log','core_mae','core_count','core_recent'}
        if set(self.parts)!=expected:raise ValueError('Incomplete all40 ensemble')
        if self.params.get('serving_profile')!='all40_v1' or self.recipe!=self.baseline.recipe:
            raise ValueError('All40 recipe differs from baseline')
        if self.params['all40_blend']!=dict(count_branch=.75,refreshed_branch=.25,count_min_mean=.5,
                count_max_mean=10.,round_int=True,count_decoder_multiplier=3.):raise ValueError('All40 mixing policy differs')
        for part in self.parts.values():part.check_schema()

    def predict(self,features):
        # Training transforms converted numerical inputs to float32 for all branches.
        frame=features.copy()
        for c in frame.select_dtypes(include=[np.number]):frame[c]=frame[c].astype('float32')
        base=self.baseline.predict(features)
        r=features.mean_28.to_numpy(float)*7;mask=(r>=.5)&(r<10)
        left=base.copy();left[mask]=self.parts['count_fine'].predict(frame)[mask]
        p={name:part.predict(frame) for name,part in self.parts.items() if name!='count_fine'}
        values={'dynamics':dict(probability=p['core_event'],magnitude=p['core_positive'],lgbm_raw=p['core_log']),
            'mae':dict(raw=p['core_mae']),'metric_count_lambda3':dict(raw=p['core_count']),
            'metric_regular_recent_mae':dict(raw=p['core_recent'])}
        refreshed=recipe_prediction(values,frame,self.recipe)
        return np.rint(.75*left+.25*refreshed)

    @property
    def selected_iterations(self):
        return {name:part.booster.current_iteration() for name,part in self.parts.items()}
