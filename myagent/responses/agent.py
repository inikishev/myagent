import json
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, Literal

import openai
import openai.types.chat
import openai.types.responses
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from loguru import logger

from .callbacks import Callback
from . import items


class BaseLanguageModel(ABC):
    """Abstract base class for language model wrappers.

    All language model implementations must inherit from this class and
    implement the `invoke` method (and `invoke_streaming` if supported).
    """

    @abstractmethod
    def invoke(
        self,
        input: list[items.BaseItem],
        tools: list[BaseTool],
        callbacks: list[Callback],
        **generation_kwargs: Any,
    ) -> list[items.BaseItem]:
        """Generate a response from the language model.

        Args:
            input: Sequence of messages.
            tools: List of tools available to the model.
            callbacks: Callback instances.
            **generation_kwargs: Additional arguments passed to the model.

        Returns:
            AssistantMessage containing the model's response.
        """
        raise NotImplementedError


    def invoke_streaming(
        self,
        input: list[items.BaseItem],
        tools: list[BaseTool],
        callbacks: list[Callback],
        **generation_kwargs: Any,
    ) -> list[items.BaseItem]:
        """Generate a streaming response from the model.

        Args:
            input: Sequence of messages.
            tools: List of tools available to the model.
            callbacks: Callback instances.
            **generation_kwargs: Additional generation parameters.

        Returns:
            AssistantMessage assembled from streamed chunks.
        """
        raise NotImplementedError

def _to_openai_responses_tool_param(tool: BaseTool) -> openai.types.responses.tool_param.FunctionToolParam:
    spec = convert_to_openai_tool(tool)
    function_spec: dict = spec["function"]
    return openai.types.responses.tool_param.FunctionToolParam(
        name = function_spec["name"],
        description = function_spec["description"],
        parameters = function_spec["parameters"],
        type = spec["type"],
        strict = True,
    )

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
        input: list[items.BaseItem],
        tools: list[BaseTool],
        callbacks: list[Callback],
        model: str | None = None,
        **generation_kwargs: Any,
    ) -> list[items.BaseItem]:
        generation_kwargs.update(self.generation_kwargs)
        if model is not None:
            generation_kwargs["model"] = model

        # Generate a response
        response: openai.types.responses.Response = self.client.responses.create(
            input=cast(Any, input),
            tools=[_to_openai_responses_tool_param(t) for t in tools],
            **generation_kwargs,
        )

        # TODO: support callbacks
        outputs = [items.to_item(item.model_dump()) for item in response.output]
        if outputs:
            outputs[0].extra_metadata["raw"] = response
        return outputs

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


def ensure_list[T](x: T | Sequence[T] | None) -> list[T]:
    if x is None: return []
    if isinstance(x, Sequence): return list(x)
    return [x]


def agent_step(
    lm: AnyLanguageModel,
    input: items.AnyItem | Sequence[items.AnyItem],
    tools: BaseTool | Sequence[BaseTool] | None = None,
    streaming: bool = False,
    callbacks: Callback | Sequence[Callback] | None = None,
    **generation_kwargs: Any,
) -> list[items.BaseItem]:
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

    input = [items.to_item(i) for i in ensure_list(input)]
    callbacks = ensure_list(callbacks)
    tools = ensure_list(tools)

    if streaming:
        return lm.invoke_streaming(
            messages=input,
            tools=tools,
            callbacks=callbacks,
            **generation_kwargs,
        )

    else:
        return lm.invoke(
            input=input,
            tools=tools,
            callbacks=callbacks,
            **generation_kwargs,
        )


def run_agent(
    lm: AnyLanguageModel,
    input: items.AnyItem | Sequence[items.AnyItem],
    tools: BaseTool | Sequence[BaseTool] | None = None,
    streaming: bool = False,
    callbacks: Callback | Sequence[Callback] | None = None,
    catch_tool_exceptions: bool = False,
    missing_strategy: Literal["skip", "message", "raise"] = "message",
    **generation_kwargs: Any,
) -> list[items.BaseItem]:
    lm = get_lm(lm)
    input = [items.to_item(i) for i in ensure_list(input)]
    callbacks = ensure_list(callbacks)
    tools = ensure_list(tools)

    for cb in callbacks:
        cb.on_agent_start(input=input, tools=tools, callbacks=callbacks, lm=lm, streaming=streaming)

    step = 0
    while True:
        response = None

        for cb in callbacks:
            cb.on_step_start(step=step, messages=input, tools=tools, callbacks=callbacks, lm=lm, streaming=streaming)

        # Get a non-empty response from the model
        response = agent_step(
            lm=lm,
            input=input,
            tools=tools,
            streaming=streaming,
            callbacks=callbacks,
            **generation_kwargs,
        )

        assert response is not None
        for cb in callbacks:
            cb.on_response_received(response)

        input.extend(response)
        
        # Invoke tools
        tool_outputs = items.invoke_tools(
            response,
            tools=tools,
            catch_exceptions=catch_tool_exceptions,
            callbacks=callbacks,
            missing_strategy=missing_strategy,
        )

        input.extend(tool_outputs)

        for cb in callbacks:
            cb.on_step_end(step=step, messages=input, tools=tools, callbacks=callbacks, lm=lm, streaming=streaming)

        step += 1

        if len(tool_outputs) == 0:
            # no tool outputs, this is the final message
            break

    for cb in callbacks:
        cb.on_agent_end(messages=input, tools=tools, callbacks=callbacks, lm=lm, streaming=streaming)

    return input
