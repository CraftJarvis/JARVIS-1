"""Exercise provider selection and the pinned SDK's serialized HTTP requests."""

import importlib
import json

import httpx
import openai
import pytest


@pytest.fixture
def provider(monkeypatch):
    for name in (
        "MINIMAX_BASE_URL", "MINIMAX_REGION", "MINIMAX_MODEL", "OPENAI_API_KEY",
        "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-test-key")
    requests = []
    clients = []
    real_client = openai.OpenAI

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Action: 2",
                "reasoning_content": "Choose the equipped tool.",
            }}],
        })

    def make_client(**kwargs):
        client = real_client(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", make_client)
    from jarvis.assembly import base, core
    importlib.reload(base)
    importlib.reload(core)
    yield base, core, requests
    for client in clients:
        client.close()


def planner_request(base, core, requests, monkeypatch):
    core.client = base.build_client()
    choices = [{"text": "dig down"}, {"text": "equip pickaxe"}]
    monkeypatch.setitem(core.skills, "test_ore", choices)
    info = {
        "inventory": [{"type": "none", "quantity": 0}] * 36,
        "equipped_items": {"mainhand": {"type": "air"}},
        "player_pos": {"y": 60},
    }
    assert core.get_skill("test_ore", info) == choices[1]
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert [m["role"] for m in payload["messages"]] == [
        "system", "user", "assistant", "user", "assistant", "user", "assistant", "user",
    ]
    assert payload["messages"][2]["content"].endswith("Action: 2")
    assert "Obtain test_ore" in payload["messages"][-1]["content"]
    for unsupported in ("thinking", "reasoning_effort", "reasoning_split", "service_tier"):
        assert unsupported not in payload
    return payload


@pytest.mark.parametrize("region,host", [
    (None, "api.minimax.io"), ("global_en", "api.minimax.io"),
    ("cn_zh", "api.minimax.cn"),
])
def test_default_planner_http_request(provider, monkeypatch, region, host):
    base, core, requests = provider
    monkeypatch.setenv("OPENAI_API_KEY", "original-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://original.example.test/v1")
    if region:
        monkeypatch.setenv("MINIMAX_REGION", region)
    assert base.MINIMAX_MODELS == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    assert core.JARVIS().model == "MiniMax-M3.1-Flash"
    payload = planner_request(base, core, requests, monkeypatch)
    assert payload["model"] == "MiniMax-M3.1-Flash"
    assert payload["max_tokens"] == 16384
    assert requests[0].url.host == host
    assert requests[0].headers["authorization"] == "Bearer minimax-test-key"


@pytest.mark.parametrize("model,budget", [
    ("MiniMax-M3", 16384), ("MiniMax-M2.7", 256), ("custom-saved-model", 256),
])
def test_explicit_model_reaches_planner_http(provider, monkeypatch, model, budget):
    base, core, requests = provider
    monkeypatch.setenv("MINIMAX_MODEL", model)
    assert core.JARVIS().model == model
    assert core.JARVIS(model="saved-constructor-model").model == "saved-constructor-model"
    payload = planner_request(base, core, requests, monkeypatch)
    assert payload["model"] == model
    assert payload["max_tokens"] == budget


@pytest.mark.parametrize("host", ["custom.example.test", "api.minimaxi.com"])
def test_custom_endpoint_overrides_region(provider, monkeypatch, host):
    base, core, requests = provider
    monkeypatch.setenv("MINIMAX_REGION", "cn_zh")
    monkeypatch.setenv("MINIMAX_BASE_URL", f"https://{host}/v1")
    monkeypatch.setenv("MINIMAX_MODEL", "custom-saved-model")
    payload = planner_request(base, core, requests, monkeypatch)
    assert requests[0].url.host == host
    assert payload["model"] == "custom-saved-model"


@pytest.mark.parametrize("model,url", [
    (None, None), ("saved-original-model", "https://original.example.test/v1"),
])
def test_original_provider_is_preserved(provider, monkeypatch, model, url):
    base, core, requests = provider
    monkeypatch.delenv("MINIMAX_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "original-test-key")
    if model:
        monkeypatch.setenv("OPENAI_MODEL", model)
    if url:
        monkeypatch.setenv("OPENAI_BASE_URL", url)
    payload = planner_request(base, core, requests, monkeypatch)
    assert payload["model"] == (model or "gpt-3.5-turbo")
    assert payload["max_tokens"] == 256
    assert str(requests[0].url) == (url or "https://api.openai.com/v1") + "/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer original-test-key"


@pytest.mark.parametrize("setting,value", [
    ("MINIMAX_MODEL", "MiniMax-M3"), ("MINIMAX_REGION", "cn_zh"),
    ("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
])
def test_minimax_never_borrows_original_credential(provider, monkeypatch, setting, value):
    base, _, requests = provider
    monkeypatch.delenv("MINIMAX_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "original-test-key")
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError, match="MINIMAX_API_KEY is required"):
        base.build_client()
    assert not requests


@pytest.mark.parametrize("model,thinking", [
    ("MiniMax-M3.1-Flash", "adaptive"),
    ("MiniMax-M3", "adaptive"), ("MiniMax-M3", "disabled"),
])
def test_sdk_serializes_supported_thinking(provider, model, thinking):
    base, _, requests = provider
    base.build_client().chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Choose a skill"}],
        extra_body={"thinking": {"type": thinking}},
    )
    assert json.loads(requests[0].content)["thinking"] == {"type": thinking}


@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "xhigh", "max"])
def test_sdk_preserves_effort_tools_reasoning_history_and_images(provider, model, effort):
    # The planner itself only sends text. This checks that its configured SDK
    # transport preserves richer messages for callers; it is not a tool loop.
    base, _, requests = provider
    client = base.build_client()
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Inspect the inventory"},
            {"type": "image_url", "image_url": {"url": "https://example.test/frame.png"}},
        ]},
        {"role": "assistant", "content": "<think>Inspect inventory first.</think>",
         "reasoning_content": "Inspect inventory first.", "tool_calls": [
             {"id": "call_inventory", "type": "function", "function": {
                 "name": "inventory", "arguments": "{}",
             }},
         ]},
        {"role": "tool", "tool_call_id": "call_inventory", "content": '{"iron": 1}'},
    ]
    tools = [{"type": "function", "function": {
        "name": "inventory", "parameters": {"type": "object", "properties": {}},
    }}]
    # openai==1.16.0 predates the named reasoning_effort argument.
    extra = {} if effort is None else {"reasoning_effort": effort}
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools, extra_body=extra,
    )
    payload = json.loads(requests[0].content)
    assert payload["messages"] == messages
    assert payload["tools"] == tools
    assert payload["model"] == model
    assert payload.get("reasoning_effort") == effort
    assert not {"thinking", "reasoning_split", "service_tier"} & payload.keys()
    assert response.choices[0].message.reasoning_content == "Choose the equipped tool."
    history = messages + [response.choices[0].message.model_dump(exclude_none=True)]
    client.chat.completions.create(model=model, messages=history)
    assert json.loads(requests[1].content)["messages"] == history
