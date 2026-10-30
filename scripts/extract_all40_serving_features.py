"""One-time extraction of the verified experiment transforms into runtime code."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'train_all_improvements.py').read_text(encoding='utf-8')
tree=ast.parse(source)
history=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='History')
method=next(n for n in history.body if isinstance(n,ast.FunctionDef) and n.name=='add')
lines=source.splitlines();add='\n'.join(lines[method.lineno-1:method.end_lineno])
event_source=(ROOT/'scripts/pycharm_event_round09.py').read_text(encoding='utf-8')
event_tree=ast.parse(event_source)
event=next(n for n in event_tree.body if isinstance(n,ast.FunctionDef) and n.name=='extra_features')
extra='\n'.join(event_source.splitlines()[event.lineno-1:event.end_lineno])
schema=json.loads((ROOT/'artifacts/all40-0a385edab6b1/models/count_fine/FINAL/refit_schema.json').read_text(encoding='utf-8'))
header='''"""Past-only serving transforms, identical to the verified all40 experiment.

History comes exclusively from the current request; no local data files or
experiment modules are used during inference. Missing history stays unknown.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import KEY

'''
constructor='''class AdvancedHistory:
    def __init__(self,sales,stock,cutoff):
        cutoff=pd.Timestamp(cutoff)
        s=sales.loc[pd.to_datetime(sales.date)<=cutoff,['date',*KEY,'quantity']].copy()
        s['date']=pd.to_datetime(s.date);s['quantity']=s.quantity.astype('float32')
        for k in KEY:s[k]=s[k].astype(str).astype('category')
        self.first=s.groupby(KEY,observed=True).date.min()
        self.first.index=self.first.index.set_levels([self.first.index.levels[i].astype(str) for i in range(2)])
        self.start=s.date.min();s['week']=s.date-pd.to_timedelta(s.date.dt.weekday,unit='D')
        self.weekly=s.groupby([*KEY,'week'],observed=True).quantity.sum().unstack('week')
        self.daily=s.groupby([*KEY,'date'],observed=True).quantity.sum().unstack('date')
        article='char_article' if 'char_article' in stock.columns else 'article'
        st=stock.loc[pd.to_datetime(stock.date)<=cutoff,['date','warehouse_name',article,'stock']].copy()
        st=st.rename(columns={article:'article'});st['date']=pd.to_datetime(st.date);st['stock']=st.stock.astype('float32')
        for k in KEY:st[k]=st[k].astype(str).astype('category')
        self.stock=st.groupby([*KEY,'date'],observed=True).stock.sum().unstack('date')
        for table in [self.weekly,self.daily,self.stock]:
            table.index=table.index.set_levels([table.index.levels[i].astype(str) for i in range(2)])

'''
tail='''

def advanced_spec():
    return dict(version=1,history_complete_weeks=52,unknown_history='nan_with_support',
                source='request_only_as_of_cutoff',feature_columns=ADVANCED_COLUMNS)


def augment_advanced_features(sales,stock,frame,cutoff):
    result=AdvancedHistory(sales,stock,cutoff).add(frame,cutoff)
    extra=extra_features(result)
    for column in extra:
        if column not in result:result[column]=extra[column].to_numpy()
    for column in result.select_dtypes(include=[np.number]):result[column]=result[column].astype('float32')
    missing=set(ADVANCED_COLUMNS)-set(result)
    if missing:raise ValueError('Missing advanced feature columns: '+repr(sorted(missing)))
    return result
'''
(ROOT/'app/advanced_features.py').write_text(header+'ADVANCED_COLUMNS = '+repr(schema['columns'])+'\n\n\n'+constructor+add+'\n\n\n'+extra+tail,encoding='utf-8')
print('Extracted serving features:',len(schema['columns']))
