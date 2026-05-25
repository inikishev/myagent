"""Message types. They are is JSON-serializeable and OpenAI-compliant and
can be for example passed directly to `ChatOpenAI.client.chat.completions`.
Anything that `chat.completions` doesn't use is stored under `extra_metadata` key."""

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import pydantic
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from .callbacks import Callback

class BaseMessage(dict[str, Any]):
    """Base class for all message types in agent conversations.

    Extends dict because UserDict is not JSON-serializeable for some reason.
    The dict schema follows OpenAI API exactly."""

    def __init__(self, role: str, content: str | None):
        """Initialize a base message.

        Args:
            role: The role of the message sender (e.g., 'system', 'user', 'assistant', 'tool').
            content: The text content of the message, or None if empty.
        """
        super().__init__(role=role, content=content, extra_metadata={})

    @classmethod
    def from_dict(cls, d: dict):
        """Creates this object from an OpenAI-compliant dictionary."""
        return cls(d["role"], d["content"])

    @property
    def role(self) -> str:
        """The role of the message sender."""
        return self["role"]

    @role.setter
    def role(self, value: str) -> None:
        self["role"] = value

    @property
    def content(self) -> str | None:
        """The text content of the message."""
        return self["content"]

    @content.setter
    def content(self, value: str | None) -> None:
        self["content"] = value

    @property
    def extra_metadata(self) -> dict:
        """Additional data not used by OpenAI API."""
        return self["extra_metadata"]

    @extra_metadata.setter
    def extra_metadata(self, value: dict) -> None:
        self["extra_metadata"] = value

    def __repr__(self) -> str:
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.items())
        return f"{self.__class__.__name__}({kwargs_str})"

class SystemMessage(BaseMessage):
    """A message from the system, typically containing instructions or context for the agent."""

    def __init__(self, content: str | None):
        """Initialize a system message.

        Args:
            content: The system prompt or instruction content.
        """
        super().__init__("system", content)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["content"])

class UserMessage(BaseMessage):
    """A message from the user, representing input or queries to the agent."""

    def __init__(self, content: str | None):
        """Initialize a user message.

        Args:
            content: The user's input or query content.
        """
        super().__init__("user", content)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["content"])

class ToolMessage(BaseMessage):
    """A message containing the result of a tool execution.

    Represents the output returned by a tool after being invoked by the agent.
    """

    def __init__(self, tool_call_id: str, content: str | None = None, tool_call: "ToolCall | None" = None):
        """Initialize a tool message.

        Args:
            tool_call_id: Unique identifier linking this message to the original tool call.
            content: The result or error message from tool execution.
            tool_call: Tool call that created this tool message. This argument is not used by the agent.
                This argument is automatically passed by `agent_step` and `run_agent`, but may be omitted
                when creating a message manually.
        """
        super().__init__("tool", content)
        self["tool_call_id"] = tool_call_id
        if tool_call is not None:
            self.extra_metadata["tool_call"] = tool_call

    @classmethod
    def from_dict(cls, d: dict):
        tc = None
        tc_dict: dict | None = d.get("extra_metatata", {}).get("tool_call", None)
        if tc_dict is not None:
            if all(isinstance(tc_dict.get(key, None), str) in tc_dict for key in ["id", "name", "arguments"]):
                tc = ToolCall(id=tc_dict["id"], name=tc_dict["name"], arguments=tc_dict["arguments"])

        return cls(tool_call_id=d["tool_call_id"], content=d["content"], tool_call=tc)

    @property
    def tool_call_id(self) -> str:
        """The unique identifier of the tool call this message responds to."""
        return self["tool_call_id"]

    @tool_call_id.setter
    def tool_call_id(self, value: str) -> None:
        self["tool_call_id"] = value

    @property
    def tool_call(self) -> "ToolCall | None":
        """`ToolCall` object that created this tool message."""
        return self.extra_metadata.get("tool_call", None)

    @tool_call.setter
    def tool_call(self, value: "ToolCall | None") -> None:
        self.extra_metadata["tool_call"] = value


class ToolCall(dict[str, Any]):
    """Represents a request from the assistant to call a tool.

    Contains the tool name, arguments, and a unique identifier for tracking
    the call and its corresponding ToolMessage response.
    """

    def __init__(self, id: str, name: str, arguments: str):
        """Initialize a tool call.

        Args:
            id: Unique identifier for this tool call.
            name: The name of the tool to invoke.
            arguments: JSON string of arguments to pass to the tool.
        """
        super().__init__(id=id, function={"name": name, "arguments": arguments}, type="function")


    @classmethod
    def from_dict(cls, d: dict):
        return cls(id=d["id"], name=d["function"]["name"], arguments=d["function"]["arguments"])

    @property
    def id(self) -> str:
        """Unique identifier for this tool call."""
        return self["id"]

    @id.setter
    def id(self, value: str) -> None:
        self["id"] = value

    @property
    def function(self) -> dict:
        """Function dictionary containing `name` and `arguments`."""
        return self["function"]

    @function.setter
    def function(self, value: dict) -> None:
        self["function"] = value

    @property
    def name(self) -> str:
        """Name of the tool to invoke."""
        return self.function["name"]

    @name.setter
    def name(self, value: str) -> None:
        self.function["name"] = value

    @property
    def arguments(self) -> str:
        """JSON string of arguments to pass to the tool."""
        return self.function["arguments"]

    @arguments.setter
    def arguments(self, value: str) -> None:
        self.function["arguments"] = value

    def parse_arguments(self) -> dict:
        """Parse the JSON arguments string into a dictionary.

        Returns:
            Dictionary of parsed arguments.

        Raises:
            json.JSONDecodeError: If arguments is not valid JSON.
        """
        return json.loads(self.arguments)

    def invoke_tool(self, tool: BaseTool, catch_exceptions: bool = True) -> "ToolMessage":
        """Execute the tool with parsed arguments and return a ToolMessage with the result.

        Args:
            tool: The Tool instance to invoke.
            catch_exceptions: If True, catch exceptions and return error in ToolMessage content.

        Returns:
            ToolMessage containing the tool execution result or error message.
        """
        try:
            result = tool.invoke(self.parse_arguments())
            content = str(result)

        except pydantic.ValidationError as e:
            content = f"ERROR: arguments for tool `{self.name}` are incorrect:\n{e}"

        except Exception if catch_exceptions else () as e:
            content = f"Exception while calling tool `{self.name}`:\n{e}"

        return ToolMessage(tool_call_id=self.id, content=content, tool_call=self)

    def __repr__(self) -> str:
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.items())
        return f"{self.__class__.__name__}({kwargs_str})"

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class AssistantMessage(BaseMessage):
    """A message from the assistant, potentially containing tool calls and reasoning.

    Represents the model's response, which may include text content, tool call requests,
    reasoning content (for models that support it), and a finish reason.
    """

    def __init__(
        self,
        content: str | None,
        tool_calls: Sequence[ToolCall] | None = None,
        finish_reason: FinishReason = "stop",
        reasoning: str | None = None,
    ):
        """Initialize an assistant message.

        Args:
            content: Text response from the model, or None if only tool calls.
            tool_calls: List of tool call requests from the model.
            finish_reason: Why the model stopped generating (e.g., 'stop', 'tool_calls').
            reasoning: Chain-of-thought reasoning content, if supported by the model.
        """
        super().__init__("assistant", content)
        if tool_calls is None:
            tool_calls = []
        else:
            tool_calls = list(tool_calls)

        self["tool_calls"] = tool_calls

        self.extra_metadata["finish_reason"] = finish_reason
        self.extra_metadata["reasoning"] = reasoning

    @classmethod
    def from_dict(cls, d: dict):
        tool_calls = [ToolCall.from_dict(tc) for tc in d.get("tool_calls", [])]
        obj = cls(content=d["content"], tool_calls=tool_calls)
        obj.extra_metadata = d.get("extra_metadata", {})
        return obj

    @property
    def finish_reason(self) -> FinishReason:
        """Why the model stopped generating."""
        return self.extra_metadata["finish_reason"]

    @finish_reason.setter
    def finish_reason(self, value: FinishReason) -> None:
        self.extra_metadata["finish_reason"] = value

    @property
    def reasoning(self) -> str | None:
        """Chain-of-thought reasoning content, if supported by the model."""
        return self.extra_metadata["reasoning"]

    @reasoning.setter
    def reasoning(self, value: str | None) -> None:
        self.extra_metadata["reasoning"] = value

    @property
    def tool_calls(self) -> list[ToolCall]:
        """List of tool call requests from the model."""
        return self["tool_calls"]

    @tool_calls.setter
    def tool_calls(self, value: list[ToolCall]) -> None:
        self["tool_calls"] = value

    def invoke_tools(
        self, tools: BaseTool | Sequence[BaseTool] | None, catch_exceptions: bool = False, callbacks: "list[Callback] | None" = None,
    ) -> list[ToolMessage]:
        """Execute all tools requested by the model and return ToolMessages.

        Args:
            tools: Available tools to invoke. Can be a single Tool or a sequence.
            catch_exceptions: If True, catch exceptions and place error messages in ToolMessage content.
            callbacks: Callbacks

        Returns:
            List of ToolMessages containing results from each tool invocation.

        Raises:
            AssertionError: If tool_calls exist but no tools are provided.
            KeyError: If a requested tool name is not found in the provided tools.
        """
        if len(self.tool_calls) == 0:
            return []

        assert tools is not None
        if isinstance(tools, BaseTool):
            tools = [tools]

        tools_dict = {tool.name: tool for tool in tools}

        tool_messages = []
        for tool_call in self.tool_calls:
            tool_message = tool_call.invoke_tool(tools_dict[tool_call.name], catch_exceptions=catch_exceptions)
            tool_messages.append(tool_message)
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_tool_return(tool_message)

        return tool_messages



AnyMessage = BaseMessage | dict[str, Any]
"""Type alias for any valid message type (BaseMessage instance or raw dictionary)."""

@overload
def to_message[T: BaseMessage](message: T) -> T: ...
@overload
def to_message(message: dict[str, Any]) -> BaseMessage: ...
def to_message(message: AnyMessage):
    """Ensures BaseMessage or one of the BaseMessage subclasses."""
    if isinstance(message, BaseMessage): return message

    role = message["role"]
    if role == "system": return SystemMessage.from_dict(message)
    if role == "user": return UserMessage.from_dict(message)
    if role == "assistant": return AssistantMessage.from_dict(message)
    if role == "tool": return ToolMessage.from_dict(message)
    return BaseMessage.from_dict(message)
