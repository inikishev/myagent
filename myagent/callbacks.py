import sys
from abc import ABC
from typing import TYPE_CHECKING

from .messages import BaseMessage, ToolCall, ToolMessage, AssistantMessage
from .tools import Tool

if TYPE_CHECKING:
    from .agent import BaseLanguageModel


class Callback(ABC):
    """Base class for agent callbacks.

    Subclass and override methods to handle specific agent events.
    All methods are no-op by default.
    """

    def on_content_token(self, token: str, lm: "BaseLanguageModel") -> None:
        """Called for each content token during streaming."""

    def on_reasoning_token(self, token: str, lm: "BaseLanguageModel") -> None:
        """Called for each reasoning token during streaming."""

    def on_tool_call_start(self, name: str, lm: "BaseLanguageModel") -> None:
        """Called when a tool call begins (name received) during streaming."""

    def on_tool_call_token(self, token: str, lm: "BaseLanguageModel") -> None:
        """Called for each tool call argument token during streaming."""

    def on_tool_call(self, tool_call: ToolCall, lm: "BaseLanguageModel") -> None:
        """Called when a tool call is fully assembled."""

    def on_tool_return(self, tool_message: ToolMessage) -> None:
        """Called when a tool execution returns a result and a ToolMessage is created."""

    def on_response_received(self, response: AssistantMessage) -> None:
        """Called when a response is received from the agent."""

    def on_agent_start(self, messages: list[BaseMessage], tools: list[Tool], callbacks: "list[Callback]", lm: "BaseLanguageModel", streaming: bool) -> None:
        """Called when the agent loop begins."""

    def on_agent_end(self, messages: list[BaseMessage], tools: list[Tool], callbacks: "list[Callback]", lm: "BaseLanguageModel", streaming: bool) -> None:
        """Called when the agent loop completes."""

    def on_step_start(self, step: int, messages: list[BaseMessage], tools: list[Tool], callbacks: "list[Callback]", lm: "BaseLanguageModel", streaming: bool) -> None:
        """Called at the start of each agent step."""

    def on_step_end(self, step: int, messages: list[BaseMessage], tools: list[Tool], callbacks: "list[Callback]", lm: "BaseLanguageModel", streaming: bool) -> None:
        """Called at the end of each agent step."""


class StreamingPrintCallback(Callback):
    """Callback that prints model output in real time with nice formatting.

    During streaming, prints reasoning content, text content, and tool calls
    as they are generated token by token. During non-streaming, prints the
    assembled response via on_response_received.

    Uses ANSI colors for readability: reasoning in gray italic, content in
    default, tool calls in cyan, arguments in yellow, results in gray.
    Pass ``color=False`` to disable ANSI codes (e.g. when logging to a file).
    """

    def __init__(self, color: bool = True) -> None:
        self._streaming = False
        self._color = color
        self._seen_reasoning = False
        self._step = 0

    def _style(self, code: str, text: str) -> str:
        if not self._color or not sys.stdout.isatty():
            return text
        return f"\033[{code}m{text}\033[0m"

    def on_agent_start(
        self,
        messages: list[BaseMessage],
        tools: list[Tool],
        callbacks: "list[Callback]",
        lm: "BaseLanguageModel",
        streaming: bool,
    ) -> None:
        self._streaming = streaming
        self._step = 0

    def on_step_start(
        self,
        step: int,
        messages: list[BaseMessage],
        tools: list[Tool],
        callbacks: "list[Callback]",
        lm: "BaseLanguageModel",
        streaming: bool,
    ) -> None:
        self._step = step
        self._seen_reasoning = False
        print(f"\n{self._style('1;33', f'── Step {step} ──')}")

    def on_reasoning_token(self, token: str, lm: "BaseLanguageModel") -> None:
        if not self._seen_reasoning:
            print(f"\n{self._style('3;90', 'Thinking...')}", end='', flush=True)
            self._seen_reasoning = True
        print(self._style('3;90', token), end='', flush=True)

    def on_content_token(self, token: str, lm: "BaseLanguageModel") -> None:
        print(token, end='', flush=True)

    def on_tool_call_start(self, name: str, lm: "BaseLanguageModel") -> None:
        print(f"\n\n{self._style('1;36', f'  ⚡ {name}(')}", end='', flush=True)

    def on_tool_call_token(self, token: str, lm: "BaseLanguageModel") -> None:
        print(self._style('93', token), end='', flush=True)

    def on_tool_call(self, tool_call: ToolCall, lm: "BaseLanguageModel") -> None:
        if self._streaming:
            print(self._style('1;36', ')'), flush=True)

    def on_tool_return(self, tool_message: ToolMessage) -> None:
        content = tool_message.content or ""
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"{self._style('90', f'  → {content}')}")

    def on_response_received(self, response: AssistantMessage) -> None:
        if not self._streaming:
            if response.reasoning:
                print(f"\n{self._style('3;90', response.reasoning)}")
            if response.content:
                print(response.content)
            if response.tool_calls:
                for tc in response.tool_calls:
                    print(f"{self._style('1;36', f'  ⚡ {tc.name}({tc.arguments})')}")
