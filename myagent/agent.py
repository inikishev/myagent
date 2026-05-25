from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import openai
import openai.types.chat
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from loguru import logger

from .callbacks import Callback
from .messages import (
    AnyMessage,
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
    to_message,
)


class BaseLanguageModel(ABC):
    """Abstract base class for language model wrappers.

    All language model implementations must inherit from this class and
    implement the `invoke` method (and `invoke_streaming` if supported).
    """

    @abstractmethod
    def invoke(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool],
        callbacks: list[Callback],
        **generation_kwargs: Any,
    ) -> AssistantMessage:
        """Generate a response from the language model.

        Args:
            messages: Sequence of conversation messages.
            tools: List of tools available to the model.
            callbacks: Callback instances.
            **generation_kwargs: Additional arguments passed to the model.

        Returns:
            AssistantMessage containing the model's response.
        """
        raise NotImplementedError


    def invoke_streaming(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool],
        callbacks: list[Callback],
        **generation_kwargs: Any,
    ) -> AssistantMessage:
        """Generate a streaming response from the model.

        Args:
            messages: Conversation history as a sequence of messages.
            tools: List of tools available to the model.
            callbacks: Callback instances.
            **generation_kwargs: Additional generation parameters.

        Returns:
            AssistantMessage assembled from streamed chunks.
        """
        raise NotImplementedError

class OpenAIWrapper(BaseLanguageModel):
    """Language model wrapper for OpenAI-compatible APIs."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str | None = None,
        **generation_kwargs: Any,
    ):

        self.client = client
        self.generation_kwargs = generation_kwargs
        if model is not None:
            self.generation_kwargs["model"] = model

    @classmethod
    def from_args(
        cls,
        base_url: str,
        model: str | None = None,
        api_key: str | Callable[[], str] | None = None,
        timeout: float | openai.Timeout | None | openai.NotGiven = openai.not_given,
        max_retries: int = openai.DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        client_kwargs: dict[str, Any] | None = None,
        **generation_kwargs: Any,
    ) -> "OpenAIWrapper":
        """Create an OpenAIWrapper from connection parameters.

        Args:
            base_url: API base URL.
            model: Model identifier to use.
            api_key: API key or callable that returns an API key.
            timeout: Request timeout configuration.
            max_retries: Maximum number of retries for failed requests.
            default_headers: Default HTTP headers to include.
            client_kwargs: Additional arguments passed to the OpenAI client.
            **generation_kwargs: Default generation parameters.

        Returns:
            Configured OpenAIWrapper instance.
        """
        if client_kwargs is None:
            client_kwargs = {}

        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            **client_kwargs,
        )

        return cls(client, model=model, **generation_kwargs)

    def invoke(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool],
        callbacks: list[Callback],
        model: str | None = None,
        **generation_kwargs: Any,
    ) -> AssistantMessage:
        generation_kwargs.update(self.generation_kwargs)
        if model is not None:
            generation_kwargs["model"] = model

        # Generate a response
        response: openai.types.chat.ChatCompletion = self.client.chat.completions.create(
            messages=cast(Any, messages),
            tools=[cast(Any, convert_to_openai_tool(t)) for t in tools],
            **generation_kwargs,
        )

        openai_assistant_msg = response.choices[0].message
        tool_calls: list[ToolCall] = []

        # Tool calls
        if openai_assistant_msg.tool_calls is not None:
            for openai_tool_call in openai_assistant_msg.tool_calls:
                assert isinstance(openai_tool_call, openai.types.chat.ChatCompletionMessageFunctionToolCall)

                tc = ToolCall(
                    id=openai_tool_call.id,
                    name=openai_tool_call.function.name,
                    arguments=openai_tool_call.function.arguments,
                )
                tool_calls.append(tc)

                for cb in callbacks:
                    cb.on_tool_call(tool_call=tc, lm=self)

        return AssistantMessage(
            content=openai_assistant_msg.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
            reasoning=getattr(response.choices[0].message, "reasoning_content", None),
        )

    def invoke_streaming(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool],
        callbacks: list[Callback],
        **generation_kwargs: Any,
    ) -> AssistantMessage:

        generation_kwargs.update(self.generation_kwargs)

        stream = self.client.chat.completions.create(
            messages=cast(Any, messages),
            tools=[cast(Any, convert_to_openai_tool(t)) for t in tools],
            stream=True,
            **generation_kwargs,
        )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, Any]] = {}
        finish_reason = None

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                content_parts.append(delta.content)
                for cb in callbacks:
                    cb.on_content_token(delta.content, lm=self)

            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                reasoning_parts.append(reasoning_content)
                for cb in callbacks:
                    cb.on_reasoning_token(reasoning_content, lm=self)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {"id": tc.id, "name": "", "arguments": ""}
                        if tc.function and tc.function.name:
                            for cb in callbacks:
                                cb.on_tool_call_start(tc.function.name, lm=self)

                    if tc.id:
                        tool_calls_map[idx]["id"] = tc.id

                    if tc.function:
                        if tc.function.name:
                            tool_calls_map[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_map[idx]["arguments"] += tc.function.arguments
                            for cb in callbacks:
                                cb.on_tool_call_token(tc.function.arguments, lm=self)

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        tool_calls: list[ToolCall] = []

        for idx in sorted(tool_calls_map.keys()):
            tc_data = tool_calls_map[idx]
            tc = ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=tc_data["arguments"],)
            tool_calls.append(tc)
            for cb in callbacks:
                cb.on_tool_call(tool_call=tc, lm=self)

        return AssistantMessage(
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
        )


AnyLanguageModel = BaseLanguageModel | openai.OpenAI
"""Type alias for any supported language model type."""


def get_lm(lm: AnyLanguageModel) -> BaseLanguageModel:
    """Convert a language model to a BaseLanguageModel instance.

    Args:
        lm: A BaseLanguageModel or raw OpenAI client.

    Raises:
        NotImplementedError: If the language model type is not supported.
    """
    if isinstance(lm, BaseLanguageModel):
        return lm
    if isinstance(lm, openai.OpenAI):
        return OpenAIWrapper(lm)
    raise NotImplementedError(type(lm))


def _ensure_list[T](x: T | Sequence[T] | None) -> list[T]:
    if x is None: return []
    if isinstance(x, Sequence): return list(x)
    return [x]


def agent_step(
    lm: AnyLanguageModel,
    messages: AnyMessage | Sequence[AnyMessage],
    tools: BaseTool | Sequence[BaseTool] | None = None,
    streaming: bool = False,
    callbacks: Callback | Sequence[Callback] | None = None,
    **generation_kwargs: Any,
) -> AssistantMessage:
    """Execute a single step of the agent (one model invocation).

    Args:
        lm: Language model to use.
        messages: Current conversation messages.
        tools: Tools available to the model.
        streaming: If True, use streaming mode.
        callbacks: Callback instances for agent events.
        **generation_kwargs: Additional generation parameters.

    Returns:
        AssistantMessage from the model.
    """
    lm = get_lm(lm)

    messages = [to_message(m) for m in _ensure_list(messages)]
    callbacks = _ensure_list(callbacks)
    tools = _ensure_list(tools)

    if streaming:
        return lm.invoke_streaming(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            **generation_kwargs,
        )

    else:
        return lm.invoke(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            **generation_kwargs,
        )


def _is_empty_stop_message(response: AssistantMessage) -> bool:
    """Check if the response is an empty stop message (no content and no tool calls).
    Usually indicates that something is wrong with the output.

    Args:
        response: The assistant message to check.

    Returns:
        True if the message has no content and no tool calls.
    """
    if response.tool_calls:
        return False
    if response.content:
        return False
    return True

def run_agent(
    lm: AnyLanguageModel,
    messages: AnyMessage | Sequence[AnyMessage],
    tools: BaseTool | Sequence[BaseTool] | None = None,
    empty_stop_message_retries: int = 10,
    streaming: bool = False,
    callbacks: Callback | Sequence[Callback] | None = None,
    **generation_kwargs: Any,
) -> list[BaseMessage]:
    """Run the full agent loop until the model stops.

    The agent alternates between generating responses and executing tool calls
    until the model indicates it's done (finish_reason='stop').

    Args:
        lm: Language model to use.
        messages: Initial conversation messages.
        tools: Tools available to the model.
        empty_stop_message_retries: Maximum retries for empty stop messages.
        streaming: If True, use streaming mode.
        callbacks: Callback instances for agent events.
        **generation_kwargs: Additional generation parameters.

    Returns:
        Complete conversation history including all messages and tool results.
    """
    lm = get_lm(lm)
    messages = [to_message(m) for m in _ensure_list(messages)]
    callbacks = _ensure_list(callbacks)
    tools = _ensure_list(tools)

    for cb in callbacks:
        cb.on_agent_start(messages=messages, tools=tools, callbacks=callbacks, lm=lm, streaming=streaming)

    step = 0
    while True:
        response = None

        for cb in callbacks:
            cb.on_step_start(step=step, messages=messages, tools=tools, callbacks=callbacks, lm=lm, streaming=streaming)

        # Get a non-empty response from the model
        for _ in range(empty_stop_message_retries):
            response = agent_step(
                lm=lm,
                messages=messages,
                tools=tools,
                streaming=streaming,
                callbacks=callbacks,
                **generation_kwargs,
            )
            if _is_empty_stop_message(response):
                logger.warning(f"Empty stop message ({_} / {empty_stop_message_retries})")
            else:
                break

        assert response is not None
        for cb in callbacks:
            cb.on_response_received(response)

        # Invoke tools
        tool_messages = response.invoke_tools(tools=tools, callbacks=callbacks)
        messages.append(response)
        messages.extend(tool_messages)

        for cb in callbacks:
            cb.on_step_end(step=step, messages=messages, tools=tools, callbacks=callbacks, lm=lm, streaming=streaming)

        step += 1

        if response.finish_reason == "stop":
            break

    for cb in callbacks:
        cb.on_agent_end(messages=messages, tools=tools, callbacks=callbacks, lm=lm, streaming=streaming)

    return messages

