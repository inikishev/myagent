"""myagent - A lightweight ReAct agent framework for tool-using language models.

Provides message types, tool definitions, an OpenAI-compatible wrapper,
and a ReAct agent loop that alternates between reasoning and tool execution.
"""

from langchain_core.tools import tool

from .agent import (
    AnyLanguageModel,
    BaseLanguageModel,
    OpenAIWrapper,
    agent_step,
    run_agent,
)
from .callbacks import Callback, StreamingPrintCallback
from .messages import (
    AnyMessage,
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
