from pathlib import Path

from scripts.package_release import release_files


def test_release_inventory_excludes_local_and_raw_data():
    root = Path(__file__).resolve().parents[1]
    names = {p.relative_to(root).as_posix() for p in release_files(root)}
    assert {"model/model.pkl", "forecast.csv", "tests/test_pipeline_parity.py"} <= names
    assert all(not n.startswith(("artifacts/", ".audit-tmp/", ".git/")) for n in names)
    assert "scripts/analyze_smape.py" not in names
    assert all(not n.startswith("data/") or n.split("/")[-1].startswith("sample") for n in names)
