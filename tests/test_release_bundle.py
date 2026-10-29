import json
from pathlib import Path

import pytest

from scripts.verify_release import verify

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_release_and_tampered_manifest(tmp_path):
    paths = [ROOT / "forecast.csv", ROOT / "required_pairs.csv", ROOT / "forecast.manifest.json", ROOT / "model"]
    result = verify(*paths)
    assert result["rows"] == 79101
    assert not result["external_key_equality_asserted_by_this_check"]
    manifest = json.loads(paths[2].read_text(encoding="utf-8"))
    manifest["forecast_sha256"] = "0" * 64
    tampered = tmp_path / "manifest.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="forecast_sha256"):
        verify(paths[0], paths[1], tampered, paths[3])
    manifest["forecast_sha256"] = result["forecast_sha256"]
    manifest["first_week"] = "2026-07-20"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="first_week"):
        verify(paths[0], paths[1], tampered, paths[3])
