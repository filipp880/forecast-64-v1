from pathlib import Path
import pytest

from scripts.promote_cv_release import publish_files


def test_promotion_check_failure_restores_all_original_files(tmp_path):
    first,second=tmp_path/'model.pkl',tmp_path/'forecast.csv'
    first.write_bytes(b'old-model');second.write_bytes(b'old-csv')
    def fail():raise ValueError('Invalid published release')
    with pytest.raises(ValueError,match='Invalid published'):
        publish_files({first:b'new-model',second:b'new-csv'},tmp_path,tmp_path/'backup',fail)
    assert first.read_bytes()==b'old-model' and second.read_bytes()==b'old-csv'
    assert (tmp_path/'backup/0.before').read_bytes()==b'old-model'


def test_promotion_rejects_paths_outside_workspace(tmp_path):
    with pytest.raises(ValueError,match='escapes'):
        publish_files({tmp_path.parent/'outside.pkl':b'data'},tmp_path,tmp_path/'backup')
