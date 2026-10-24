from __future__ import annotations

import argparse

import pytest
from fle import cli, config
from fle.util import FleError


def write_conf(tmp_path, text):
    path = tmp_path / "conf.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def ns(path, **kw):
    base = {"config": path, "profile": None, "spec": None, "engine": None, "host": None, "port": None,
            "context": None, "threads": None, "gpu_cache_mib": None, "fit_target_mib": None,
            "pinned_load": None, "no_mtp": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_defaults_when_nothing_is_set(tmp_path):
    s = cli._settings(ns(str(tmp_path / "missing.toml")))
    assert s["profile"] == "auto" and s["port"] == 8080 and s["mtp"] is True and s["spec"] == "auto"


def test_file_beats_defaults(tmp_path):
    path = write_conf(tmp_path, 'profile = "balanced"\nport = 9000\nmtp = false\n[env]\nFLE_A = "1"\n')
    s = cli._settings(ns(path))
    assert (s["profile"], s["port"], s["mtp"]) == ("balanced", 9000, False)
    assert s["env"] == {"FLE_A": "1"}


def test_environment_beats_file(tmp_path, monkeypatch):
    path = write_conf(tmp_path, 'profile = "balanced"\nport = 9000\n')
    monkeypatch.setenv("FLE_PROFILE", "max")
    monkeypatch.setenv("FLE_PORT", "9100")
    monkeypatch.setenv("FLE_MTP", "0")
    s = cli._settings(ns(path))
    assert (s["profile"], s["port"], s["mtp"]) == ("max", 9100, False)


def test_cli_beats_environment_and_file(tmp_path, monkeypatch):
    path = write_conf(tmp_path, 'profile = "balanced"\nport = 9000\n')
    monkeypatch.setenv("FLE_PROFILE", "max")
    monkeypatch.setenv("FLE_PORT", "9100")
    s = cli._settings(ns(path, profile="nvme", port=9200))
    assert (s["profile"], s["port"]) == ("nvme", 9200)


def test_cli_none_does_not_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FLE_SPEC", "ngram-mod")
    s = cli._settings(ns(str(tmp_path / "none.toml"), spec=None))
    assert s["spec"] == "ngram-mod"


def test_no_mtp_flag_wins_over_everything(tmp_path, monkeypatch):
    path = write_conf(tmp_path, "mtp = true\n")
    monkeypatch.setenv("FLE_MTP", "1")
    assert cli._settings(ns(path, no_mtp=True))["mtp"] is False


def test_env_dicts_merge_in_precedence_order():
    out = config.effective({"env": {"B": "cli"}}, {"env": {"A": "file", "B": "file"}})
    assert out["env"] == {"A": "file", "B": "cli"}


def test_unknown_profile_is_rejected_everywhere(tmp_path, monkeypatch):
    with pytest.raises(FleError, match="Unknown profile"):
        cli._settings(ns(write_conf(tmp_path, 'profile = "turbo"\n')))
    monkeypatch.setenv("FLE_PROFILE", "turbo")
    with pytest.raises(FleError, match="Unknown profile"):
        cli._settings(ns(str(tmp_path / "none.toml")))


def test_unknown_keys_in_the_file_are_ignored(tmp_path):
    s = cli._settings(ns(write_conf(tmp_path, 'future_key = 1\nport = 7\n')))
    assert s["port"] == 7 and "future_key" not in s


def test_config_path_order(tmp_path, monkeypatch):
    assert config.config_path("/x/y.toml").as_posix() == "/x/y.toml"
    monkeypatch.setenv("FLE_CONFIG", str(tmp_path / "from-env.toml"))
    assert config.config_path().name == "from-env.toml"
    monkeypatch.delenv("FLE_CONFIG")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg2"))
    assert config.config_path().as_posix().endswith("xdg2/fle/config.toml")


def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "rt.toml")
    data, _ = config.load(path)
    data.update({"profile": "nvme", "port": 1234, "mtp": False, "engine": "/e/llama-server"})
    config.set_dotted(data, "env.FLE_MC_OVERLAP", "2")
    config.save(data, path)
    back, _ = config.load(path)
    assert (back["profile"], back["port"], back["mtp"], back["engine"]) == ("nvme", 1234, False, "/e/llama-server")
    assert back["env"] == {"FLE_MC_OVERLAP": "2"}


def test_fallback_toml_parser_matches_for_our_keys():
    text = 'profile = "max"\nport = 9\nmtp = false # comment\n[env]\nFLE_X = "y"\n'
    assert config._parse_fallback(text) == {"profile": "max", "port": 9, "mtp": False, "env": {"FLE_X": "y"}}


def test_dotted_helpers():
    data = {"a": {"b": 1}}
    assert config.get_dotted(data, "a.b") == 1
    config.set_dotted(data, "a.c.d", 2)
    assert data["a"]["c"] == {"d": 2}
    config.unset_dotted(data, "a.b")
    assert "b" not in data["a"]
    with pytest.raises(KeyError):
        config.get_dotted(data, "a.zzz")
