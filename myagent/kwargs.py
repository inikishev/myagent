"""kwargs suggested by developers of those models"""
import copy
from collections import UserDict
from typing import Self


class KwargsDict(UserDict):
    def override(self, **kwargs) -> Self:
        d = copy.deepcopy(self)
        d.update(kwargs)
        return d

    def without_keys(self, *keys: str) -> Self:
        d = copy.deepcopy(self)
        for k in keys: d.pop(k, None)
        return d

# we make those classes to get autocomplete
class Qwen3_5(KwargsDict):
    think_general=KwargsDict({
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        },
    })

    think_coding=KwargsDict({
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        },
    })

    instruct_general=KwargsDict({
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    })

    instruct_reasoning=KwargsDict({
        "temperature": 1.0,
        "top_p": 1.0,
        "presence_penalty": 2.0,
        "extra_body": {
            "top_k": 40,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    })

class Qwen3_6:
    think_general=KwargsDict({
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        },
    })

    think_coding=KwargsDict({
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        },
    })

    instruct=KwargsDict({
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    })

class Gemma4:
    think=KwargsDict({
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 64,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        },
    })

    think_preserve=KwargsDict({
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 64,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        },
    })

    instruct=KwargsDict({
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 64,
            "chat_template_kwargs": {"enable_thinking": False,},
        },
    })


class Nanbeige4_1:
    general=KwargsDict({
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {
            "repetition_penalty": 1.0,
        },
    })

class LFM2_5_8BA1B:
    general=KwargsDict({
        "temperature": 0.2,
        "extra_body": {
            "top_k": 80,
            "repetition_penalty": 1.05,
        },
    })

class LFM2_5_1_2B:
    general=KwargsDict({
        "temperature": 0.1,
        "extra_body": {
            "top_k": 50,
            "repetition_penalty": 1.05,
        },
    })

class MiniCPM5:
    think=KwargsDict({
        "temperature": 0.9,
        "top_p": 0.95,
        "extra_body": {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": True,},
        },
    })

    instruct=KwargsDict({
        "temperature": 0.7,
        "top_p": 0.95,
        "extra_body": {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False,},
        },
    })

class Granite4_1:
    general=KwargsDict({
        "temperature": 0.2,
        "top_p": 1.0,
        "extra_body": {
            "top_k": 0.0,
        },
    })

class Nemotron3Nano:
    think_reasoning = KwargsDict({
        "temperature": 1.0,
        "top_p": 1.0,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True,},
        },
    })
    think_tool_calling = KwargsDict({
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True,},
        },
    })
    instruct_reasoning = KwargsDict({
        "temperature": 1.0,
        "top_p": 1.0,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False,},
        },
    })
    instruct_tool_calling = KwargsDict({
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False,},
        },
    })

class Qwen3_8:
    think=KwargsDict({
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        },
    })

    instruct=KwargsDict({
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
        },
    })