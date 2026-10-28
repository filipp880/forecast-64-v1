"""Serializable candidates and forward-selected ensemble for ideas.txt."""
from __future__ import annotations
import copy
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .base import BaseForecastModel
from .blend_model import BlendForecastModel
from .research_model import member_values, recipe_prediction
from ..features import CAT_FEATURES, KEY
from ..enhancements import feature_columns
from ..ideas_features import ideas_columns, ideas_spec

POST = dict(medium_gate=.5, regular_scale=1.1, round_int=True, round_threshold=.5)


def columns_for(spec):
    return feature_columns('dynamics')+ideas_columns(spec.get('groups', []))


def blend_params(spec, threads=6, estimators=600):
    parts = spec.get('regularization', {})
    return dict(feature_columns=columns_for(spec), threads=threads, seed=42, weight_hurdle=.3,
                hurdle=dict(log_period=0,
                            classifier=dict(n_estimators=estimators, **parts.get('classifier', {})),
                            regressor=dict(n_estimators=estimators, **parts.get('regressor', {}))),
                lgbm=dict(n_estimators=estimators, log_period=0, **parts.get('lgbm', {})))


class HorizonPart:
    """Global heads pooling all pairs; either 9 horizons or three 3-week bands."""
    def __init__(self, spec, threads=6, estimators=600):
        self.spec, self.threads, self.estimators = spec, threads, estimators
        self.models = {}

    def groups(self, frame):
        return ((frame.h.to_numpy(int)-1)//int(self.spec.get('head_width', 9)))

    def fit(self, train, valid=None, iterations=None):
        groups = self.groups(train)
        vg = self.groups(valid) if valid is not None else None
        self.models = {}
        for group in sorted(set(groups)):
            mask = groups==group
            model = BlendForecastModel(blend_params(self.spec, self.threads, self.estimators))
            v = valid.loc[vg==group] if valid is not None else None
            model.fit(train.loc[mask], train.target.to_numpy(float)[mask], v,
                      v.target.to_numpy(float) if v is not None else None, 60,
                      n_iterations=iterations.get(str(group)) if iterations else None)
            self.models[str(group)] = model
        return self

    @property
    def selected_iterations(self): return {key: m.selected_iterations for key,m in self.models.items()}

    def predict(self, frame):
        prediction = np.zeros(len(frame))
        groups = self.groups(frame)
        for key, model in self.models.items():
            mask = groups==int(key)
            if mask.any():
                v = member_values(model, frame.loc[mask])
                # Keep continuous base forecasts; ensemble rounds only once.
                post = {**POST, 'round_int': False}
                prediction[mask] = recipe_prediction({'model':v}, frame.loc[mask], {'source':'model','post':post})
        return prediction


class XGBPart:
    def __init__(self, spec, threads=6, estimators=600):
        self.spec, self.threads, self.estimators = spec, threads, estimators
        self.categories = {}

    def encode(self, frame, fit=False):
        x = frame[columns_for(self.spec)].copy()
        for col in CAT_FEATURES:
            if fit: self.categories[col] = sorted(x[col].astype(str).unique())
            x[col] = pd.Categorical(x[col].astype(str), categories=self.categories[col])
        return x

    def fit(self, train, valid=None, iterations=None):
        from xgboost import XGBRegressor
        params = dict(n_estimators=int(iterations or self.estimators), max_depth=6, learning_rate=.05,
                      tree_method='hist', enable_categorical=True, objective='reg:squarederror',
                      reg_alpha=1., reg_lambda=12., colsample_bytree=.8, subsample=.9,
                      n_jobs=self.threads, random_state=42, eval_metric='rmse')
        if valid is not None and iterations is None: params['early_stopping_rounds'] = 60
        self.model = XGBRegressor(**params)
        x = self.encode(train, True)
        kwargs = dict(eval_set=[(self.encode(valid), np.log1p(valid.target.to_numpy(float)))], verbose=False) if valid is not None else {}
        self.model.fit(x, np.log1p(train.target.to_numpy(float)), **kwargs)
        self.iterations = int(self.model.best_iteration+1) if valid is not None and iterations is None else int(iterations or self.estimators)
        return self

    @property
    def selected_iterations(self): return self.iterations

    def predict(self, frame): return np.maximum(np.expm1(self.model.predict(self.encode(frame))), 0.)


def _network(sizes, future_width):
    import torch.nn as nn
    return nn.ModuleDict({
        'embeddings': nn.ModuleList([nn.Embedding(n+1, 6, padding_idx=0) for n in sizes]),
        'conv': nn.ModuleList([nn.Conv1d(3 if i==0 else 16, 16, 2, dilation=2**i) for i in range(4)]),
        'dense': nn.Sequential(nn.Linear(16+3*13+18+future_width, 64), nn.ReLU(), nn.Linear(64,9))})


def _forward(net, seq, cat, future):
    import torch
    import torch.nn.functional as F
    value = seq
    for i, conv in enumerate(net['conv']): value = F.relu(conv(F.pad(value, (2**i,0))))
    emb = torch.cat([e(cat[:,i]) for i,e in enumerate(net['embeddings'])], dim=1)
    return net['dense'](torch.cat([value[:,:,-1], seq.flatten(1), emb, future], dim=1))


class CNNPart:
    """Small CPU global causal CNN; shared 9-week output, masked partial labels."""
    CATEGORIES = ['warehouse_name','article','category_lvl_1']

    def __init__(self, spec, threads=6, estimators=600):
        self.spec, self.threads = spec, threads
        self.categories = {}

    def arrays(self, frame, fit=False):
        from ..ideas_features import event_calendar, EVENTS
        group = ['feature_origin', *KEY]
        first = frame.drop_duplicates(group).copy()
        index = pd.MultiIndex.from_frame(first[group])
        seq = np.stack([first[[f'weekly_lag_{i}' for i in range(13,0,-1)]].to_numpy(float),
                        first[[f'article_weekly_lag_{i}' for i in range(13,0,-1)]].to_numpy(float)], axis=1)
        mask = np.isfinite(seq[:,0,:]).astype(float)
        seq = np.nan_to_num(np.log1p(np.maximum(seq,0)), nan=0.)
        if fit: self.scale = max(float(seq.std()), .1)
        seq = np.concatenate([seq/self.scale, mask[:,None,:]], axis=1).astype('float32')
        cat = np.zeros((len(first),3), dtype='int64')
        for i,col in enumerate(self.CATEGORIES):
            if fit: self.categories[col] = {s:j+1 for j,s in enumerate(sorted(first[col].astype(str).unique()))}
            cat[:,i] = first[col].astype(str).map(self.categories[col]).fillna(0).to_numpy(int)
        # Build all known future dates even for origins with incomplete labels.
        # Respect explicit first_week too, while h still identifies the head.
        start = pd.to_datetime(first.week)-pd.to_timedelta((first.h.to_numpy(int)-1)*7,unit='D')
        weeks = start.to_numpy()[:,None]+np.arange(9)[None,:]*np.timedelta64(7,'D')
        cal = event_calendar(weeks.ravel())
        selected = ['month_has_15','month_has_end','month_begin_days','month_end_days',
                    *[f'{name}_{kind}' for name in EVENTS for kind in ('during','pre_week','post_week')]]
        future = cal.reindex(weeks.ravel())[selected].to_numpy('float32').reshape(len(first), -1)/7
        target = np.zeros((len(first),9), dtype='float32'); known = np.zeros_like(target)
        if 'target' in frame:
            for h in range(1,10):
                part = frame.loc[frame.h.eq(h)].set_index(group).target.reindex(index)
                known[:,h-1] = part.notna()
                target[:,h-1] = np.log1p(part.fillna(0).to_numpy(float))
        return (seq, cat, future, target, known), index

    def fit(self, train, valid=None, iterations=None):
        import torch
        torch.set_num_threads(self.threads); torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        arrays,_ = self.arrays(train, True)
        tensors = [torch.from_numpy(a) for a in arrays]
        validation = [torch.from_numpy(a) for a in self.arrays(valid)[0]] if valid is not None else None
        self.sizes = [len(self.categories[c]) for c in self.CATEGORIES]
        self.future_width = arrays[2].shape[1]
        net = _network(self.sizes,self.future_width)
        optimizer = torch.optim.Adam(net.parameters(), lr=.002)
        best, stale, chosen = float('inf'), 0, 1
        epochs = int(iterations or 18)
        best_state = None
        generator = torch.Generator().manual_seed(42)
        for epoch in range(epochs):
            net.train()
            for ids in torch.randperm(len(arrays[0]), generator=generator).split(1024):
                s,c,f,y,m = [a[ids] for a in tensors]
                optimizer.zero_grad(); p = _forward(net,s,c,f)
                loss = (((p-y)**2)*m).sum()/m.sum().clamp_min(1)
                loss.backward(); optimizer.step()
            if validation is not None and iterations is None:
                net.eval(); total,count = 0.,0.
                with torch.no_grad():
                    for ids in torch.arange(len(validation[0])).split(2048):
                        s,c,f,y,m = [a[ids] for a in validation]
                        total += float((((_forward(net,s,c,f)-y)**2)*m).sum()); count += float(m.sum())
                score = total/max(count,1)
                if score < best-1e-7:
                    best,stale,chosen = score,0,epoch+1; best_state = copy.deepcopy(net.state_dict())
                else: stale += 1
                if stale>=4: break
            else: chosen = epoch+1
        if best_state is not None: net.load_state_dict(best_state)
        self.state = {k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
        self.iterations = chosen
        return self

    @property
    def selected_iterations(self): return self.iterations

    def predict(self, frame):
        import torch
        torch.set_num_threads(self.threads)
        arrays,index = self.arrays(frame)
        net = _network(self.sizes,self.future_width)
        net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in self.state.items()}); net.eval()
        result=[]
        with torch.no_grad():
            for ids in np.array_split(np.arange(len(arrays[0])), max(1,int(np.ceil(len(arrays[0])/2048)))):
                s,c,f = [torch.from_numpy(a[ids]) for a in arrays[:3]]
                result.append(np.maximum(np.expm1(np.clip(_forward(net,s,c,f).numpy(),0,15)),0))
        predictions = np.concatenate(result)
        codes = index.get_indexer(pd.MultiIndex.from_frame(frame[['feature_origin',*KEY]]))
        if (codes<0).any(): raise ValueError('CNN pair alignment failed')
        return predictions[codes,frame.h.to_numpy(int)-1]


def fit_ideas(spec, train, valid=None, iterations=None, threads=6, estimators=600):
    kind = spec.get('kind','blend')
    if kind=='xgb': return XGBPart(spec,threads,estimators).fit(train,valid,iterations)
    if kind=='cnn': return CNNPart(spec,threads,estimators).fit(train,valid,iterations)
    return HorizonPart(spec,threads,estimators).fit(train,valid,iterations)


class WeeklyIdeasModel(BaseForecastModel):
    name='ideas'; file_suffix='.pkl'; target_transform='none'

    def __init__(self, params=None):
        super().__init__(params); self.members=[]

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, early_stopping_rounds=None, n_iterations=None):
        train = x_train.assign(target=y_train)
        valid = x_valid.assign(target=y_valid) if x_valid is not None else None
        self.members = [fit_ideas(s,train,valid,(n_iterations or [None]*len(self.params['specs']))[i],
                                 self.params.get('threads',6),self.params.get('estimators',600))
                        for i,s in enumerate(self.params['specs'])]
        return self

    @property
    def selected_iterations(self): return [m.selected_iterations for m in self.members]

    def predict(self, features):
        if 'feature_origin' not in features: raise ValueError('Missing feature origin')
        raw = sum(w*m.predict(features) for w,m in zip(self.params['weights'],self.members))
        return np.rint(np.maximum(raw,0.))

    def save(self,path): Path(path).write_bytes(pickle.dumps(self,protocol=4))

    @classmethod
    def load(cls,path):
        obj=pickle.loads(Path(path).read_bytes())
        if not isinstance(obj,cls): raise ValueError('Invalid ideas model')
        obj.check_schema(); return obj

    def check_schema(self):
        if len(self.members)!=len(self.params['specs']) or len(self.members)!=len(self.params['weights']): raise ValueError('Ideas members mismatch')
        if not np.isclose(sum(self.params['weights']),1) or min(self.params['weights'])<0: raise ValueError('Invalid ideas weights')
        for spec,part in zip(self.params['specs'],self.members):
            if part.spec!=spec: raise ValueError('Ideas member spec mismatch')
            if isinstance(part,HorizonPart):
                for model in part.models.values():
                    for booster in (model.hurdle.classifier,model.hurdle.regressor,model.lgbm.booster):
                        if booster.feature_name()!=columns_for(spec): raise ValueError('Ideas booster schema mismatch')
            if isinstance(part,XGBPart) and part.model.get_booster().feature_names!=columns_for(spec): raise ValueError('Ideas XGB schema mismatch')
