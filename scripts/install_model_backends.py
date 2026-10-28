"""Install only optional CPU backends actually required by the packaged model."""
import json
import subprocess
import sys
from pathlib import Path

config=json.loads(Path('model/config.json').read_text(encoding='utf-8'))
kinds={s.get('kind','blend') for s in config.get('model_params',{}).get('specs',[])} if config.get('model_type')=='ideas' else set()
if 'xgb' in kinds:
    subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir','xgboost==3.2.0'])
if 'cnn' in kinds:
    subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir','torch==2.13.0',
                           '--index-url','https://download.pytorch.org/whl/cpu'])
