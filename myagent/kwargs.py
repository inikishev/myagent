"""kwargs suggested by developers of those models"""

# we make those classes to get autocomplete
class Qwen3_5:
    think_general={
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        },
    }
    
    think_coding={
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        },
    }
    
    instruct_general={
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }

    instruct_reasoning={
        "temperature": 1.0,
        "top_p": 1.0,
        "presence_penalty": 2.0,
        "extra_body": {
            "top_k": 40,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }

class Qwen3_6:
    think_general={
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        },
    }
    
    think_coding={
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        },
    }
    
    instruct={
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.0,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }

class Gemma4:
    think={
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 64,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        },
    }
    
    instruct={
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 64,
            "chat_template_kwargs": {"enable_thinking": False,},
        },
    }
    