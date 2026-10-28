"""Serializable weekly research recipe; training and serving use identical code."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .base import BaseForecastModel, inverse_target, transform_target
from .blend_model import BlendForecastModel
from .lightgbm_model import LightGBMForecastModel
from ..features import CAT_FEATURES, KEY
from ..enhancements import feature_columns
from ..calibration import combine


class CatPart:
    def __init__(self, columns, threads=6):
        self.columns,self.threads=columns,threads
        self.model=None

    def encode(self,x):
        result=x[self.columns].copy()
        for c in CAT_FEATURES:result[c]=result[c].astype(str)
        return result

    def fit(self,x,y,valid=None,y_valid=None,iterations=None):
        from catboost import CatBoostRegressor
        self.model=CatBoostRegressor(iterations=int(iterations or 400),depth=6,learning_rate=.05,
            loss_function="RMSE",random_seed=42,thread_count=self.threads,verbose=False,allow_writing_files=False)
        kwargs={"cat_features":CAT_FEATURES}
        if valid is not None and iterations is None:
            kwargs.update(eval_set=(self.encode(valid),np.log1p(y_valid)),early_stopping_rounds=60)
        self.model.fit(self.encode(x),np.log1p(y),**kwargs)
        return self

    @property
    def selected_iterations(self):return self.model.tree_count_

    def predict(self,x):return np.maximum(np.expm1(self.model.predict(self.encode(x),thread_count=self.threads)),0.)


class GroupPart:
    """Four past-demand groups, each pooling items/warehouses, plus global fallback."""
    def __init__(self,params):self.params=params;self.models={}

    @staticmethod
    def groups(x):
        zero=x.zero_week_share_13.to_numpy()
        return np.where(zero>=.8,0,np.where(zero>=.4,1,np.where(x.positive_cv2_13.to_numpy()>1,2,3)))

    def fit(self,x,y,valid=None,y_valid=None,iterations=None):
        groups=self.groups(x)
        valid_groups=self.groups(valid) if valid is not None else None
        self.models={}
        for g in [-1,0,1,2,3]:
            mask=np.ones(len(x),bool) if g==-1 else groups==g
            if mask.sum()<500:continue
            params={**self.params,"num_leaves":31,"reg_lambda":12.}
            model=LightGBMForecastModel(params)
            chosen=iterations.get(str(g)) if iterations else None
            vm=np.ones(len(valid),bool) if g==-1 and valid is not None else (valid_groups==g if valid is not None else None)
            xv=valid.loc[vm] if vm is not None and vm.sum()>30 else None
            yv=np.log1p(y_valid[vm]) if xv is not None else None
            model.fit(x.loc[mask],np.log1p(y[mask]),xv,yv,60,n_iterations=chosen)
            self.models[str(g)]=model
        return self

    @property
    def selected_iterations(self):return {g:m.selected_iterations for g,m in self.models.items()}

    def predict(self,x):
        result=np.maximum(np.expm1(self.models['-1'].predict(x)),0.)
        groups=self.groups(x)
        for g,m in self.models.items():
            mask=groups==int(g)
            if g!='-1' and mask.any():result[mask]=np.maximum(np.expm1(m.predict(x.loc[mask])),0.)
        return result


def fit_member(name, x, valid=None, *, iterations=None, threads=6, estimators=600):
    profiles=['base','calendar','dynamics','monetary','all']
    profile=name if name in profiles else 'all'
    columns=feature_columns(profile)
    common=dict(feature_columns=columns,threads=threads,seed=42,n_estimators=estimators,log_period=0)
    y=x.target.to_numpy(float);yv=valid.target.to_numpy(float) if valid is not None else None
    if name in profiles:
        params={**common,"weight_hurdle":.3,"hurdle":{"log_period":0,"classifier":{"n_estimators":estimators},"regressor":{"n_estimators":estimators}},"lgbm":{"n_estimators":estimators,"log_period":0}}
        model=BlendForecastModel(params)
        model.fit(x,y,valid,yv,60,n_iterations=iterations)
    elif name=='catboost':
        model=CatPart(columns,threads).fit(x,y,valid,yv,iterations)
    elif name=='grouped':
        model=GroupPart(common).fit(x,y,valid,yv,iterations)
    else:
        objectives={'poisson':'poisson','tweedie':'tweedie','mae':'regression_l1','quantile':'quantile'}
        params={**common,"objective":objectives[name],"target_transform":"none"}
        if name=='tweedie':params['tweedie_variance_power']=1.5
        if name=='quantile':params.update(alpha=.5,eval_metric='quantile')
        if name=='mae':params['eval_metric']='l1'
        model=LightGBMForecastModel(params)
        model.fit(x,y,valid,yv,60,n_iterations=iterations)
    return model


def member_values(model,features):
    if isinstance(model,BlendForecastModel):
        x=model.hurdle._encode(features,False);threads=int(model.params.get('threads',6))
        p=model.hurdle.classifier.predict(x,num_iteration=model.hurdle._best_iterations['classifier'],num_threads=threads)
        m=model.hurdle.regressor.predict(x,num_iteration=model.hurdle._best_iterations['regressor'],num_threads=threads)
        lg=inverse_target(model.lgbm.predict(features),model.lgbm.target_transform)
        return dict(probability=p,magnitude=m,lgbm_raw=np.maximum(lg,0.))
    if isinstance(model,LightGBMForecastModel):return dict(raw=np.maximum(inverse_target(model.predict(features),model.target_transform),0.))
    return dict(raw=model.predict(features))


def quantize(raw,features,spec):
    raw=np.maximum(np.asarray(raw,float),0.)
    if spec.get('apportion'):
        codes,_=pd.factorize(pd.MultiIndex.from_frame(features[KEY]))
        if len(raw)%9==0 and np.array_equal(codes,np.repeat(np.arange(len(raw)//9),9)):
            values=raw.reshape(-1,9);floor=np.floor(values)
            count=(np.rint(values.sum(axis=1))-floor.sum(axis=1)).astype(int)
            ranks=np.argsort(np.argsort(-(values-floor),axis=1,kind='stable'),axis=1,kind='stable')
            return (floor+(ranks<count[:,None])).ravel()
        result=np.floor(raw)
        for g in range(codes.max()+1):
            idx=np.flatnonzero(codes==g)
            count=int(np.rint(raw[idx].sum())-result[idx].sum())
            order=np.argsort(-(raw[idx]-result[idx]),kind='stable')
            result[idx[order[:count]]]+=1
        return result
    if spec.get('round_int',True):
        threshold=float(spec.get('round_threshold',.5))
        return np.rint(raw) if threshold==.5 else np.floor(raw+1-threshold)
    return raw


def recipe_prediction(values,features,recipe):
    source=recipe['source'];post=recipe.get('post',{})
    if source in ('tsb','sba'):
        raw=features[source+'_week'].to_numpy(float)*post.get('scale',1.)
    elif 'raw' in values[source]:raw=values[source]['raw']*post.get('scale',1.)
    else:
        v=values[source]
        probability=v['probability'];base_post={k:v for k,v in post.items() if k not in ['apportion','round_threshold','round_int','rare_gate','weekly_regular_scale']}
        if 'rare_gate' in post:
            gate=np.where(features.zero_week_share_13.to_numpy()>=.6,post['rare_gate'],post.get('gate',.5))
            probability=(probability>=gate).astype(float)
            base_post.update(gate=.5,medium_gate=.5)
        raw=combine(probability,v['magnitude'],v['lgbm_raw']*recipe.get('smearing',1.),
                    features.mean_28.to_numpy(float)*7,features.n_sale_days_90.to_numpy(float),
                    base_post)
    if 'weekly_regular_scale' in post:
        regular=(features.zero_week_share_13.to_numpy()<.3)&(features.positive_cv2_13.to_numpy()<1)
        raw=raw*np.where(regular,post['weekly_regular_scale'],1.)
    if recipe.get('aux'):
        aux=recipe['aux'];w=recipe['aux_weight']
        extra=features[aux+'_week'].to_numpy(float) if aux in ('tsb','sba') else values[aux]['raw']
        raw=(1-w)*raw+w*extra
    return quantize(raw,features,post)


class WeeklyResearchModel(BaseForecastModel):
    name='research';file_suffix='.pkl';target_transform='none';supports_early_stopping=True

    def __init__(self,params=None):
        super().__init__(params)
        self.members={};self.recipe=self.params.get('recipe',{'source':'all','post':{'round_int':True}})

    def fit(self,x_train,y_train,x_valid=None,y_valid=None,early_stopping_rounds=None,n_iterations=None):
        train=x_train.assign(target=y_train)
        valid=x_valid.assign(target=y_valid) if x_valid is not None else None
        names={self.recipe['source'],self.recipe.get('aux')}-{'tsb','sba',None}
        for name in sorted(names):
            self.members[name]=fit_member(name,train,valid,iterations=(n_iterations or {}).get(name),
                                        threads=int(self.params.get('threads',6)),estimators=int(self.params.get('estimators',600)))
        return self

    def predict(self,features):
        values={k:member_values(v,features) for k,v in self.members.items()}
        return recipe_prediction(values,features,self.recipe)

    @property
    def selected_iterations(self):return {k:v.selected_iterations for k,v in self.members.items()}

    def save(self,path):Path(path).write_bytes(pickle.dumps(self,protocol=4))

    @classmethod
    def load(cls,path):
        model=pickle.loads(Path(path).read_bytes())
        if not isinstance(model,cls):raise ValueError('Invalid research model payload')
        if model.recipe!=model.params.get('recipe'):raise ValueError('Recipe metadata mismatch')
        return model

    def check_schema(self):
        def check(model):
            if isinstance(model,BlendForecastModel):check(model.hurdle);check(model.lgbm)
            elif isinstance(model,GroupPart):
                for m in model.models.values():check(m)
            elif isinstance(model,CatPart):
                if list(model.model.feature_names_)!=model.columns:raise ValueError('CatBoost schema mismatch')
            else:
                for attr in ('booster','classifier','regressor'):
                    booster=getattr(model,attr,None)
                    if booster is not None and booster.feature_name()!=model.params['feature_columns']:raise ValueError('Research booster schema mismatch')
        for member in self.members.values():check(member)
