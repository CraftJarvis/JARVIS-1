"""Tests for the configurable MiniMax provider setup in ``jarvis.assembly.base``."""

import importlib

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # A placeholder key lets the OpenAI-compatible client be constructed at
    # import time without contacting the network.
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    for var in ("MINIMAX_BASE_URL", "MINIMAX_REGION", "MINIMAX_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _load_base():
    import jarvis.assembly.base as base

    return importlib.reload(base)


def test_default_region_is_global():
    base = _load_base()
    assert base.DEFAULT_REGION == "global_en"
    assert base.get_base_url() == "https://api.minimax.io/v1"


def test_cn_region_uses_mainland_endpoint(monkeypatch):
    monkeypatch.setenv("MINIMAX_REGION", "cn_zh")
    base = _load_base()
    assert base.get_base_url() == "https://api.minimaxi.com/v1"


def test_explicit_base_url_overrides_region(monkeypatch):
    monkeypatch.setenv("MINIMAX_REGION", "cn_zh")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://example.test/v1")
    base = _load_base()
    assert base.get_base_url() == "https://example.test/v1"


def test_default_model():
    base = _load_base()
    assert base.MINIMAX_MODELS[:2] == ["MiniMax-M3", "MiniMax-M2.7"]
    assert base.DEFAULT_MODEL == "MiniMax-M3"
    assert base.get_model() == "MiniMax-M3"


def test_model_selection(monkeypatch):
    monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M2.7")
    base = _load_base()
    assert base.get_model() == "MiniMax-M2.7"


def test_client_points_at_resolved_base_url():
    base = _load_base()
    assert str(base.client.base_url).rstrip("/") == "https://api.minimax.io/v1"
