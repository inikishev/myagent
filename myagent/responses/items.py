import base64
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, cast, overload

import pydantic
from langchain_core.tools import BaseTool

from .callbacks import Callback


class BaseItem(dict[str, Any], ABC):
    """In responses API the chat is a list of items such as messages, tool calls, tool call outputs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault("extra_metadata", {})

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        ...

    @property
    def type(self) -> str:
        """Type of this item (message, function_call_output, etc)."""
        return self["type"]

    @type.setter
    def type(self, value: str) -> None:
        self["type"] = value

    @property
    def extra_metadata(self) -> dict:
        """Additional data not used by OpenAI API."""
        return self["extra_metadata"]

    @extra_metadata.setter
    def extra_metadata(self, value: dict) -> None:
        self["extra_metadata"] = value

    def __repr__(self) -> str:
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.items() if k != 'extra_metadata')
        return f"{self.__class__.__name__}({kwargs_str})"


class BaseContent(dict[str, Any], ABC):
    """content in a message (text, image or audio)"""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        ...

    def __repr__(self) -> str:
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.items())
        return f"{self.__class__.__name__}({kwargs_str})"

class BaseTextContent(BaseContent):
    """read the name of the class"""
    def __init__(self, text: str, type: str):
        super().__init__(text=text, type=type)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(text=d["text"], type=d["type"])

    @property
    def text(self) -> str:
        """Type of this item (message, function_call_output, etc)."""
        return self["text"]

    @text.setter
    def text(self, value: str) -> None:
        self["text"] = value

    @property
    def type(self) -> str:
        return self["type"]

    @type.setter
    def type(self, value: str) -> None:
        self["type"] = value

class InputTextContent(BaseTextContent):
    """read the name of the class"""
    def __init__(self, text):
        super().__init__(text, "input_text")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(text=d["text"])

class OutputTextContent(BaseTextContent):
    """read the name of the class"""
    def __init__(self, text):
        super().__init__(text, "output_text")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(text=d["text"])

# TODO: InputImageContent
# class ResponseInputImageParam(TypedDict, total=False):
#     """An image input to the model.

#     Learn about [image inputs](https://platform.openai.com/docs/guides/vision).
#     """

#     detail: Required[Literal["low", "high", "auto", "original"]]
#     """The detail level of the image to be sent to the model.

#     One of `high`, `low`, `auto`, or `original`. Defaults to `auto`.
#     """

#     type: Required[Literal["input_image"]]
#     """The type of the input item. Always `input_image`."""

#     file_id: Optional[str]
#     """The ID of the file to be sent to the model."""

#     image_url: Optional[str]
#     """The URL of the image to be sent to the model.

#     A fully qualified URL or base64 encoded image in a data URL.
#     """

Detail = Literal["low", "medium", "high"]

class InputImageContent(BaseContent):
    def __init__(self, image_url: str, detail: Detail = "high"):
        super().__init__(image_url=image_url, detail=detail, type="input_image")

    @classmethod
    def from_local_file(cls, path: str | os.PathLike, detail: Detail = "high"):
        with open(path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            b64 = encoded_bytes.decode("utf-8")

        mime = Path(path).suffix.lower()
        if mime == "jpg": mime = "jpeg"

        return InputImageContent(image_url=f"data:image/{mime};base64,{b64}", detail=detail)

    @classmethod
    def from_dict(cls, d):
        return cls(image_url=d["image_url"], detail=d.get("detail", "high"))

    @property
    def image_url(self) -> str:
        return self["image_url"]

    @image_url.setter
    def image_url(self, value: str) -> None:
        self["image_url"] = value

    @property
    def detail(self) -> Detail:
        """The role of the message sender."""
        return cast(Detail, self["detail"])

    @detail.setter
    def detail(self, value: Detail) -> None:
        self["detail"] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(detail={self.detail})"

Text = InputTextContent
Image = InputImageContent

AnyContent = BaseContent | dict[str, Any]

@overload
def to_content[T: BaseContent](content: T) -> T: ...
@overload
def to_content(content: dict[str, Any]) -> BaseContent: ...
def to_content(content: AnyContent):
    if isinstance(content, BaseContent): return content

    type = content["type"]
    if type == "input_text": return InputTextContent.from_dict(content)
    if type == "output_text": return InputTextContent.from_dict(content)
    return BaseTextContent.from_dict(content)

class BaseMessageItem(BaseItem):
    def __init__(self, role: Literal["user", "system", "developer", "assistant"], content: Sequence[BaseContent]):
        super().__init__(role=role, content=list(content), type="message", status="complete")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(role=d["role"], content=[to_content(c) for c in d["content"]])

    @property
    def role(self) -> Literal["user", "system", "developer", "assistant"]:
        return self["role"]

    @role.setter
    def role(self, value: Literal["user", "system", "developer", "assistant"]) -> None:
        self["role"] = value

    @property
    def content(self) -> list[BaseContent]:
        return self["content"]

    @content.setter
    def content(self, value: list[BaseContent]) -> None:
        self["content"] = value

    @property
    def status(self) -> str:
        """In openai responses its `Literal["in_progress", "completed", "incomplete"]`, idk what it does"""
        return self["status"]

    @status.setter
    def status(self, value: str) -> None:
        self["status"] = value

class InputMessageItem(BaseMessageItem):
    """Input message  (user, system or developer)."""
    def __init__(self, role: Literal["user", "system", "developer"], content: Sequence[BaseContent]):
        super().__init__(role=role, content=content)


# NOTE:
# content is a list of ResponseOutputText (or ResponseOutputRefusal)
# ResponseOutputText has those fields: `annotations`, `text`, `type` (always "output_text"), and `logprobs`
# I'm pretty sure all of those are only used by OpenAI, so we only care about `text` string

class OutputMessageItem(BaseMessageItem):
    """Output message of the model."""
    def __init__(self, id: str, content: Sequence[OutputTextContent], phase: Literal["commentary", "final_answer"] | None = None):
        super().__init__(role="assistant", content=content)
        self.id = id
        self.phase = phase

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(id=d["id"], content=[OutputTextContent.from_dict(c) for c in d["content"]], phase=d.get("phase", None))

    @property
    def id(self) -> str:
        return self["id"]

    @id.setter
    def id(self, value: str) -> None:
        self["id"] = value

    @property
    def phase(self) -> Literal["commentary", "final_answer"] | None:
        return self["phase"]

    @phase.setter
    def phase(self, value: Literal["commentary", "final_answer"] | None) -> None:
        self["phase"] = value

    def _get_output(self) -> OutputTextContent:
        # I don't know how a model could produce multiple contents, should always be 1
        assert len(self.content) == 1, self.content
        output = self.content[0]
        assert isinstance(output, OutputTextContent)
        return output

    @property
    def text(self) -> str:
        return self._get_output().text

    @text.setter
    def text(self, value: str) -> None:
        self._get_output().text = value


class BaseToolCallItem(BaseItem):
    def __init__(self, call_id: str, id: str | None, type: Literal['function_call', "function_call_output"]):
        super().__init__(call_id=call_id, id=id, type=type, status="completed")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(call_id=d["call_id"], id=d.get("id", None), type=d["type"])

    # idk the difference between call id and id
    @property
    def call_id(self) -> str:
        return self["call_id"]

    @call_id.setter
    def call_id(self, value: str) -> None:
        self["call_id"] = value

    @property
    def id(self) -> str | None:
        return self["id"]

    @id.setter
    def id(self, value: str | None) -> None:
        self["id"] = value

    @property
    def status(self) -> str:
        return self["status"]

    @status.setter
    def status(self, value: str) -> None:
        self["status"] = value

class ToolCallItem(BaseToolCallItem):
    """A tool call request made by the model.

    Args:
        call_id: unique id of the tool call
        name: name of the tool
        arguments: json string

    """
    def __init__(self, call_id: str, name: str, arguments: str, id: str | None = None):
        super().__init__(call_id=call_id, id=id, type="function_call")
        self.name = name
        self.arguments = arguments

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(call_id=d["call_id"], id=d.get("id", None), name=d["name"], arguments=d["arguments"])

    @property
    def arguments(self) -> str:
        return self["arguments"]

    @arguments.setter
    def arguments(self, value: str) -> None:
        self["arguments"] = value

    @property
    def name(self) -> str:
        return self["name"]

    @name.setter
    def name(self, value: str) -> None:
        self["name"] = value

    def parse_arguments(self) -> dict:
        return json.loads(self.arguments)


    def invoke_tool(self, tool: BaseTool, catch_exceptions: bool = False) -> "ToolOutputItem":
        try:
            result = tool.invoke(self.parse_arguments())
            # Output can be str or list of Content objects.
            if isinstance(result, BaseContent): output = [result]
            elif isinstance(result, Sequence) and len(result) > 0 and isinstance(result[0], BaseContent): output = list(result)
            else: output = str(result)

        except pydantic.ValidationError as e:
            output = f"ERROR: arguments for tool `{self.name}` are incorrect:\n{e}"

        except Exception if catch_exceptions else () as e:
            output = f"Exception while calling tool `{self.name}`:\n{e}"

        return ToolOutputItem(call_id=self.call_id, id=self.id, output=output, tool_call=self)


class ToolOutputItem(BaseToolCallItem):
    """What the tool call returned."""
    def __init__(self, call_id: str, output: str | Sequence[BaseContent], id: str | None = None, tool_call: ToolCallItem | None = None):
        super().__init__(call_id=call_id, id=id, type="function_call_output")
        self.output = output
        self.extra_metadata["tool_call"] = tool_call

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:

        tool_call=d.get("extra_metadata", {}).get("tool_call", None)

        if tool_call is not None:
            tool_call = ToolCallItem.from_dict(tool_call)

        return cls(
            call_id=d["call_id"],
            id=d.get("id", None),
            output=d["output"],
            tool_call=tool_call,
        )

    @property
    def output(self) -> str | Sequence[BaseContent]:
        return self["output"]

    @output.setter
    def output(self, value: str | Sequence[BaseContent]) -> None:
        self["output"] = value

    @property
    def tool_call(self) -> ToolCallItem | None:
        return self.extra_metadata["tool_call"]

    @tool_call.setter
    def tool_call(self, value: ToolCallItem | None) -> None:
        self.extra_metadata["tool_call"] = value


# summary is a list of models with two fields: text and type which is always "summary_text"
# content - same, type is "reasoning_text"
class ReasoningItem(BaseItem):
    def __init__(self, id: str, summary: list[str], content: list[str] | None):
        super().__init__(
            id=id,
            summary=[{"text": t, "type": "summary_text"} for t in summary],
            content=[{"text": t, "type": "reasoning_text"} for t in content] if content is not None else None,
            type="reasoning",
            encrypted_content=None, # idk what this is
            status="completed"
        )

    @classmethod
    def from_dict(cls, d: dict):
        content = d.get("content", None)
        if content is not None: content = [c["text"] for c in content]
        return cls(id=d["id"], summary=[s["text"] for s in d["summary"]], content=content)

    @property
    def summary(self) -> list[str]:
        return [s["text"] for s in self["summary"]]

    @summary.setter
    def summary(self, value: list[str]) -> None:
        self["summary"] = [{"text": t, "type": "summary_text"} for t in value]

    @property
    def content(self) -> list[str] | None:
        if self["content"] is None: return None
        return [s["text"] for s in self["content"]]

    @content.setter
    def content(self, value: list[str] | None) -> None:
        if value is None: self["content"] = value
        else:
            self["content"] = [{"text": t, "type": "reasoning_text"} for t in value]


class BaseMessage(BaseItem):
    """Message spec from chat completions API which is compatible with responses API, and is easier to write.
    This corresponds to `EasyInputMessageParam`"""
    def __init__(self, role: Literal["user", "assistant", "system", "developer"], content: str | Sequence[BaseContent] | None, phase: Literal["commentary", "final_answer"] | None = None):
        super().__init__(role=role, content=content, type="message", phase=phase)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(role=d["role"], content=d["content"], phase=d.get("phase", None))

    @property
    def role(self) -> Literal["user", "assistant", "system", "developer"]:
        return self["role"]

    @role.setter
    def role(self, value: Literal["user", "assistant", "system", "developer"]) -> None:
        self["role"] = value

    @property
    def content(self) -> str | Sequence[BaseContent] | None:
        return self["content"]

    @content.setter
    def content(self, value: str | Sequence[BaseContent] | None) -> None:
        self["content"] = value

    @property
    def phase(self) -> Literal["commentary", "final_answer"] | None:
        return self["phase"]

    @phase.setter
    def phase(self, value: Literal["commentary", "final_answer"] | None) -> None:
        self["phase"] = value

    @property
    def status(self) -> str:
        """In openai responses its `Literal["in_progress", "completed", "incomplete"]`, idk what it does"""
        return self["status"]

    @status.setter
    def status(self, value: str) -> None:
        self["status"] = value

class SystemMessage(BaseMessage):
    def __init__(self, content: str | Sequence[BaseContent] | None):
        super().__init__("system", content)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["content"])

class DeveloperMessage(BaseMessage):
    def __init__(self, content: str | Sequence[BaseContent] | None):
        super().__init__("developer", content)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["content"])

class UserMessage(BaseMessage):
    def __init__(self, content: str | Sequence[BaseContent] | None):
        super().__init__("user", content)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["content"])

class AssistantMessage(BaseMessage):
    """Input assistant message for context injection."""
    def __init__(self, content: str | Sequence[BaseContent] | None):
        super().__init__("assistant", content)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["content"])



AnyItem = BaseItem | dict[str, Any]

@overload
def to_item[T: BaseItem](item: T) -> T: ...
@overload
def to_item(item: dict[str, Any]) -> BaseItem: ...
def to_item(item: AnyItem):
    if isinstance(item, BaseItem): return item

    type = item["type"]
    if type == "message":
        if isinstance(item["content"], str):
            role = item["role"]
            if role == "system": return SystemMessage.from_dict(item)
            if role == "developer": return DeveloperMessage.from_dict(item)
            if role == "user": return UserMessage.from_dict(item)
            if role == "assistant": return AssistantMessage.from_dict(item)
            return BaseMessage.from_dict(item)

        if item["role"] == "assistant":
            return OutputMessageItem.from_dict(item)

        return InputMessageItem.from_dict(item)

    if type == "function_call": return ToolCallItem.from_dict(item)
    if type ==  "function_call_output": return ToolOutputItem.from_dict(item)
    if type == "reasoning": return ReasoningItem.from_dict(item)

    raise RuntimeError(f"Unkown type `{type}` on item {item}")


def invoke_tools(
    items: list[BaseItem],
    tools: BaseTool | Sequence[BaseTool] | None,
    catch_exceptions: bool = False,
    callbacks: "Callback | Sequence[Callback] | None" = None,
    missing_strategy: Literal["skip", "message", "raise"] = "message",
) -> list[ToolOutputItem]:
    """Execute all tools requested by the model and return a list of `ToolOutputItem`.

    Args:
        tools: Available tools to invoke. Can be a single Tool or a sequence.
        catch_exceptions: If True, catch exceptions and place error messages in ToolMessage content.
        callbacks: Callbacks

    Returns:
        List of ToolOutputItem containing results from each tool invocation.

    Raises:
        AssertionError: If tool_calls exist but no tools are provided.
        KeyError: If a requested tool name is not found in the provided tools.
    """
    tool_calls = [i for i in items if isinstance(i, ToolCallItem)]
    if len(tool_calls) == 0:
        return []

    if isinstance(callbacks, Callback):
        callbacks = [callbacks]

    assert tools is not None
    if isinstance(tools, BaseTool):
        tools = [tools]

    tools_dict = {tool.name: tool for tool in tools}

    tool_outputs = []
    for tool_call in tool_calls:

        if tool_call.name in tools_dict:
            output = tool_call.invoke_tool(tools_dict[tool_call.name], catch_exceptions=catch_exceptions)

        else:
            # Tool doesn't exist
            if missing_strategy == 'skip':
                output = None

            elif missing_strategy == 'message':

                output = ToolOutputItem(
                    call_id=tool_call.call_id,
                    id=tool_call.id,
                    output=f"ERROR: tool '{tool_call.name}' doesn't exist.",
                    tool_call=tool_call,
                )

            elif missing_strategy == 'raise':
                raise RuntimeError(f"Tool {tool_call.name} doesn't exist.")

            else:
                raise ValueError(missing_strategy)

        if output is not None:
            tool_outputs.append(output)
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_tool_return(output)

    return tool_outputs


