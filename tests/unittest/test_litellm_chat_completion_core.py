import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_handlers.litellm_helpers import _handle_streaming_response


class FakeBox:
    def __init__(self, values=None, **attrs):
        self._values = values or {}
        for key, value in attrs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return self._values.get(key, default)


class FakeSettings:
    def __init__(self, config_values=None, settings_values=None):
        self.config = FakeBox(
            config_values or {},
            reasoning_effort=None,
            ai_timeout=30,
            custom_reasoning_model=False,
            max_model_tokens=32000,
            verbosity_level=0,
            model="gpt-4o",
        )
        self.litellm = FakeBox()
        self._settings_values = settings_values or {}

    def get(self, key, default=None):
        return self._settings_values.get(key, default)


def _mock_response():
    mock = MagicMock()
    response = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    mock.__getitem__.side_effect = response.__getitem__
    mock.dict.return_value = response
    return mock


@pytest.mark.asyncio
async def test_chat_completion_passes_seed_when_temperature_is_zero(monkeypatch):
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: FakeSettings(config_values={"seed": 123}))

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()

        await handler.chat_completion(model="gpt-4o", system="sys", user="usr", temperature=0)

    assert mock_call.call_args.kwargs["seed"] == 123


@pytest.mark.asyncio
async def test_chat_completion_rejects_seed_for_claude_opus_4_8_default_temperature(monkeypatch):
    class FakeAPIError(Exception):
        def __init__(self, message, request=None, *, body=None):
            super().__init__(message)

    monkeypatch.setattr(litellm_handler, "get_settings", lambda: FakeSettings(config_values={"seed": 123}))
    monkeypatch.setattr(litellm_handler.openai, "APIError", FakeAPIError)

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        handler = litellm_handler.LiteLLMAIHandler()

        with pytest.raises(FakeAPIError) as exc_info:
            await handler.chat_completion(model="claude-opus-4-8", system="sys", user="usr")

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value) == "Seed (123) is not supported with temperature (0.2) > 0"
    assert str(exc_info.value.__cause__) == "Seed (123) is not supported with temperature (0.2) > 0"
    mock_call.assert_not_called()


@pytest.mark.asyncio
async def test_empty_stream_preserves_failure_reason():
    async def empty_stream():
        if False:
            yield None

    with pytest.raises(
        litellm_handler.openai.APIError,
        match="Empty streaming response received without proper completion",
    ):
        await _handle_streaming_response(empty_stream())


@pytest.mark.asyncio
async def test_stream_timeout_covers_response_consumption(monkeypatch):
    async def slow_stream():
        await asyncio.sleep(1)
        if False:
            yield None

    async def fake_completion(**kwargs):
        return slow_stream()

    monkeypatch.setattr(litellm_handler, "acompletion", fake_completion)
    handler = litellm_handler.LiteLLMAIHandler()

    with pytest.raises(
        litellm_handler.openai.APIError,
        match="LLM request exceeded timeout of 0.01s",
    ):
        await handler._get_completion(model="chatgpt/gpt-5.6-sol", timeout=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-4-8",
        "claude-opus-4-8",
        "vertex_ai/claude-opus-4-8",
        "bedrock/anthropic.claude-opus-4-8",
        "bedrock/global.anthropic.claude-opus-4-8",
        "bedrock/us.anthropic.claude-opus-4-8",
        "bedrock/eu.anthropic.claude-opus-4-8",
        "bedrock/au.anthropic.claude-opus-4-8",
        "bedrock/jp.anthropic.claude-opus-4-8",
    ],
)
async def test_chat_completion_strips_temperature_for_claude_opus_4_8(monkeypatch, model):
    monkeypatch.setattr(litellm_handler, "get_settings", FakeSettings)

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()

        await handler.chat_completion(model=model, system="sys", user="usr", temperature=0.2)

    assert "temperature" not in mock_call.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-5",
        "claude-sonnet-5",
        "vertex_ai/claude-sonnet-5",
        "bedrock/anthropic.claude-sonnet-5",
        "bedrock/global.anthropic.claude-sonnet-5",
        "bedrock/us.anthropic.claude-sonnet-5",
        "bedrock/au.anthropic.claude-sonnet-5",
        "bedrock/eu.anthropic.claude-sonnet-5",
        "bedrock/jp.anthropic.claude-sonnet-5",
    ],
)
async def test_chat_completion_strips_temperature_for_claude_sonnet_5(monkeypatch, model):
    monkeypatch.setattr(litellm_handler, "get_settings", FakeSettings)

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()

        await handler.chat_completion(model=model, system="sys", user="usr", temperature=0.2)

    assert "temperature" not in mock_call.call_args.kwargs


@pytest.mark.asyncio
async def test_chat_completion_does_not_use_extended_thinking_for_claude_opus_4_8(monkeypatch):
    monkeypatch.setattr(
        litellm_handler,
        "get_settings",
        lambda: FakeSettings(config_values={"enable_claude_extended_thinking": True}),
    )

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()

        await handler.chat_completion(model="claude-opus-4-8", system="sys", user="usr", temperature=0.2)

    assert "thinking" not in mock_call.call_args.kwargs
    assert "max_tokens" not in mock_call.call_args.kwargs
    assert "temperature" not in mock_call.call_args.kwargs


@pytest.mark.asyncio
async def test_chat_completion_does_not_use_extended_thinking_for_claude_sonnet_5(monkeypatch):
    monkeypatch.setattr(
        litellm_handler,
        "get_settings",
        lambda: FakeSettings(config_values={"enable_claude_extended_thinking": True}),
    )

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()

        await handler.chat_completion(model="claude-sonnet-5", system="sys", user="usr", temperature=0.2)

    assert "thinking" not in mock_call.call_args.kwargs
    assert "max_tokens" not in mock_call.call_args.kwargs
    assert "temperature" not in mock_call.call_args.kwargs


@pytest.mark.asyncio
async def test_chat_completion_combines_prompts_for_user_message_only_models(monkeypatch):
    monkeypatch.setattr(litellm_handler, "get_settings", FakeSettings)

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()
        handler.user_message_only_models = ["user-only-model"]

        await handler.chat_completion(model="user-only-model", system="sys", user="usr")

    messages = mock_call.call_args.kwargs["messages"]
    assert messages == [{"role": "user", "content": "sys\n\n\nusr"}]


@pytest.mark.asyncio
async def test_get_completion_uses_streaming_for_required_models():
    handler = litellm_handler.LiteLLMAIHandler.__new__(litellm_handler.LiteLLMAIHandler)
    handler.streaming_required_models = ["streaming-model"]

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call, \
            patch("pr_agent.algo.ai_handlers.litellm_ai_handler._handle_streaming_response",
                  new_callable=AsyncMock) as mock_stream:
        mock_call.return_value = "stream"
        mock_stream.return_value = ("streamed text", "stop")

        resp, finish_reason, response_obj = await handler._get_completion(
            model="streaming-model",
            messages=[],
        )

    assert mock_call.call_args.kwargs["stream"] is True
    assert resp == "streamed text"
    assert finish_reason == "stop"
    assert response_obj.dict()["choices"][0]["message"]["content"] == "streamed text"


@pytest.mark.asyncio
async def test_get_completion_uses_streaming_for_chatgpt_models(monkeypatch):
    monkeypatch.setenv("CHATGPT_ALLOW_DEVICE_LOGIN", "true")
    handler = litellm_handler.LiteLLMAIHandler.__new__(litellm_handler.LiteLLMAIHandler)
    handler.streaming_required_models = []

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call, \
            patch("pr_agent.algo.ai_handlers.litellm_ai_handler._handle_streaming_response",
                  new_callable=AsyncMock) as mock_stream:
        mock_call.return_value = "stream"
        mock_stream.return_value = ("streamed text", "stop")

        resp, finish_reason, response_obj = await handler._get_completion(
            model="chatgpt/gpt-5.5",
            messages=[],
        )

    assert mock_call.call_args.kwargs["stream"] is True
    assert resp == "streamed text"
    assert finish_reason == "stop"
    assert response_obj.dict()["choices"][0]["message"]["content"] == "streamed text"


@pytest.mark.asyncio
async def test_get_completion_disables_chatgpt_device_login_when_noninteractive(monkeypatch):
    authenticator_module = pytest.importorskip("litellm.llms.chatgpt.authenticator")
    common_utils = pytest.importorskip("litellm.llms.chatgpt.common_utils")
    Authenticator = authenticator_module.Authenticator
    GetAccessTokenError = common_utils.GetAccessTokenError

    original_login = Authenticator._login_device_code
    original_wait = Authenticator._wait_for_access_token
    had_marker = hasattr(Authenticator, "_pr_agent_device_login_disabled")
    original_marker = getattr(Authenticator, "_pr_agent_device_login_disabled", None)

    monkeypatch.delenv("CHATGPT_ALLOW_DEVICE_LOGIN", raising=False)
    monkeypatch.setattr(litellm_handler.sys.stdin, "isatty", lambda: False)
    if had_marker:
        delattr(Authenticator, "_pr_agent_device_login_disabled")

    handler = litellm_handler.LiteLLMAIHandler.__new__(litellm_handler.LiteLLMAIHandler)
    handler.streaming_required_models = []

    try:
        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call, \
                patch("pr_agent.algo.ai_handlers.litellm_ai_handler._handle_streaming_response",
                      new_callable=AsyncMock) as mock_stream:
            mock_call.return_value = "stream"
            mock_stream.return_value = ("streamed text", "stop")

            await handler._get_completion(
                model="chatgpt/gpt-5.5",
                messages=[],
            )

        with pytest.raises(GetAccessTokenError) as exc_info:
            Authenticator.__new__(Authenticator)._login_device_code()
        assert "interactive device login is disabled" in str(exc_info.value)
    finally:
        Authenticator._login_device_code = original_login
        Authenticator._wait_for_access_token = original_wait
        if had_marker:
            Authenticator._pr_agent_device_login_disabled = original_marker
        elif hasattr(Authenticator, "_pr_agent_device_login_disabled"):
            delattr(Authenticator, "_pr_agent_device_login_disabled")
