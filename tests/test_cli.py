import pytest

from hybrid_search.cli import main


def test_demo_is_explicitly_synthetic(capsys):
    assert main(["demo"]) == 0
    assert '"synthetic": true' in capsys.readouterr().out


def test_missing_inputs_fail_before_loading_caches(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        main(["check", "--data-dir", str(tmp_path), "--cache-dir", str(tmp_path)])
