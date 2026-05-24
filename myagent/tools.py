"""Tool definitions for agent function calling.

This module provides the Tool class and decorator for converting Python functions
into tools that can be called by language models during agent execution.
"""

import inspect
from collections.abc import Callable
from typing import Any, overload

import openai
import pydantic
from langchain_core.utils.function_calling import convert_to_openai_tool


def _get_tool_model_arg[MODEL: pydantic.BaseModel](function: Callable[[MODEL], Any]) -> type[MODEL]:
    """Extract pydantic model type from a function that takes a single BaseModel argument.

    Args:
        function: A function with exactly one parameter annotated as a pydantic BaseModel.

    Returns:
        The pydantic model class used as the function's parameter type.

    Raises:
        AssertionError: If the function doesn't have exactly one parameter.
    """
    signature = inspect.signature(function)
    assert len(signature.parameters) == 1
    return next(iter(signature.parameters.values())).annotation


class InvalidToolArgsError(Exception):
    """Raised when tool arguments fail pydantic validation."""

def is_base_model(x) -> bool:
    """Check if a type is a subclass of pydantic.BaseModel.

    Args:
        x: A type to check.

    Returns:
        True if x is a pydantic BaseModel subclass.
    """
    return isinstance(x, type) and issubclass(x, pydantic.BaseModel)


class Tool[RETURN]:
    """Wraps a Python function as a tool callable by language models.

    Supports two modes:
    1. Functions taking a single pydantic BaseModel argument for structured input
    2. Regular functions with type hints that are converted to JSON schema
    """

    tool_model: type[pydantic.BaseModel] | None
    """The pydantic model used for argument validation, if applicable."""

    def __init__(
        self,
        function: Callable[..., RETURN],
        name: str,
        description: str | None,
        strict: bool = False,
    ):
        """Initialize a Tool.

        Args:
            function: The Python function to wrap.
            name: Name of the tool as exposed to the model.
            description: Description of the tool's purpose as exposed to the model.
            strict: If True, generate strict JSON schema (not yet supported for pydantic functions).
        """
        self.function = function
        self.name = name
        if description is not None: description = description.strip()
        self.description = description

        signature = inspect.signature(function)
        if len(signature.parameters) == 1 and is_base_model(
            next(iter(signature.parameters.values())).annotation
        ):
            if strict:
                raise NotImplementedError("Strict not implemented for pydantic functions")

            self.tool_model = _get_tool_model_arg(function)
            assert self.tool_model is not None
            self.json_schema = openai.pydantic_function_tool(
                self.tool_model, name=name, description=description
            )

        else:
            self.tool_model = None
            self.json_schema = convert_to_openai_tool(function, strict=strict)

    def __call__(self, *args, **kwargs) -> RETURN:
        """Call the underlying function directly with positional/keyword arguments."""
        return self.function(*args, **kwargs)

    def call_with_arguments(self, args: dict[str, Any]) -> RETURN:
        """Call the tool with arguments provided by the model.

        Args:
            args: Dictionary of arguments from the model's tool call.

        Returns:
            Result of the function call.

        Raises:
            InvalidToolArgsError: If arguments fail pydantic validation.
        """
        if self.tool_model is None:
            return self.function(**args)

        try:
            instance = self.tool_model.model_validate(args)
        except pydantic.ValidationError as e:
            raise InvalidToolArgsError(e) from None

        return self.function(instance)

def _get_name_description(
    fn: Callable, name: str | None, description: str | None
) -> tuple[str, str | None]:
    """Extract name and description from a function if not explicitly provided.

    Args:
        fn: The function to inspect.
        name: Explicit name override, or None to use function name.
        description: Explicit description override, or None to use docstring.

    Returns:
        Tuple of (name, description).
    """
    if name is None:
        if hasattr(fn, "__name__"):
            name = fn.__name__
        else:
            name = fn.__class__.__name__

    if description is None:
        description = inspect.getdoc(fn)

    return name, description


@overload
def tool[RETURN](
    name_or_fn: Callable[..., RETURN],
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
) -> Tool[RETURN]:
    """Convert a Python function to a tool.

    Args:
        name_or_fn: Python function to convert.
        name: Name of the tool; if None, uses the function's name.
        description: Description of the tool; if None, uses the docstring.
        strict: Whether to generate a strict JSON schema.
    """


@overload
def tool[RETURN](
    name_or_fn: str,
    *,
    description: str | None = None,
    strict: bool = False,
) -> Callable[[Callable[..., RETURN]], Tool[RETURN]]:
    """Create a decorator that converts a function to a tool.

    Args:
        name_or_fn: Name for the tool.
        description: Description of the tool; if None, uses the docstring.
        strict: Whether to generate a strict JSON schema.
    """


def tool[RETURN](
    name_or_fn: Callable[..., RETURN] | str,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
) -> Tool[RETURN] | Callable[[Callable[..., RETURN]], Tool[RETURN]]:
    """Convert a function to a tool, either directly or as a decorator.

    Can be used in two ways:
    1. @tool decorator: @tool or @tool(name="custom_name", description="...")
    2. Direct call: tool(my_function)

    Args:
        name_or_fn: Either a function to wrap or a tool name (for decorator usage).
        name: Tool name override.
        description: Tool description override.
        strict: Whether to generate strict JSON schema.

    Returns:
        A Tool instance or a decorator function.
    """
    if callable(name_or_fn):
        name, description = _get_name_description(name_or_fn, name=name, description=description)
        return Tool(name_or_fn, name=name, description=description, strict=strict)

    def tool_wrapped[RETURN_WRAPPED](fn: Callable[..., RETURN_WRAPPED], /):
        nonlocal name, description
        name, description = _get_name_description(fn, name=name, description=description)
        return Tool(fn, name=name, description=description, strict=strict)

    return tool_wrapped

