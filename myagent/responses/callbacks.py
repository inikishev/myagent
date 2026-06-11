import sys
from abc import ABC
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from . import agent, items

class Callback(ABC):
    """Base class for agent callbacks.

    Subclass and override methods to handle specific agent events.
    All methods are no-op by default.
    """
    def on_content_token(self, token: str, lm: "agent.BaseLanguageModel") -> None:
        """Called for each content token during streaming."""

    def on_reasoning_token(self, token: str, lm: "agent.BaseLanguageModel") -> None:
        """Called for each reasoning token during streaming."""

    def on_tool_call_start(self, name: str, lm: "agent.BaseLanguageModel") -> None:
        """Called when a tool call begins (name received) during streaming."""

    def on_tool_call_token(self, token: str, lm: "agent.BaseLanguageModel") -> None:
        """Called for each tool call argument token during streaming."""

    def on_tool_call(self, tool_call: "items.ToolCallItem", lm: "agent.BaseLanguageModel") -> None:
        """Called when a tool call is fully assembled."""

    def on_tool_return(self, tool_message: "items.ToolOutputItem") -> None:
        """Called when a tool execution returns a result and a ToolCallOutputItem is created."""

    def on_response_received(self, response: "list[items.BaseItem]") -> None:
        """Called when a response is received from the agent."""

    def on_agent_start(self, input: "list[items.BaseItem]", tools: "list[BaseTool]", callbacks: "list[Callback]", lm: "agent.BaseLanguageModel", streaming: bool) -> None:
        """Called when the agent loop begins."""

    def on_agent_end(self, messages: "list[items.BaseItem]", tools: list[BaseTool], callbacks: "list[Callback]", lm: "agent.BaseLanguageModel", streaming: bool) -> None:
        """Called when the agent loop completes."""

    def on_step_start(self, step: int, messages: "list[items.BaseItem]", tools: "list[BaseTool]", callbacks: "list[Callback]", lm: "agent.BaseLanguageModel", streaming: bool) -> None:
        """Called at the start of each agent step."""

    def on_step_end(self, step: int, messages: "list[items.BaseItem]", tools: "list[BaseTool]", callbacks: "list[Callback]", lm: "agent.BaseLanguageModel", streaming: bool) -> None:
        """Called at the end of each agent step."""
