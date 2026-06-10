# update20260610
Im going to make a new one for openai responses API and it will be in a new repo and this one is done for

<h1 align='center'>myagent</h1>

Very simple abstraction for OpenAI chat completions API (technically can be used for other ones like `transformers` but ive not implemented it yet).

I became lazy after writing the sentence above. So no more readme for now.

Alright alright I'll put the example here.


```python
from openai import OpenAI
import myagent

client = OpenAI(
    base_url="http://127.0.0.1:8888/v1",
    api_key="sk-unsloth-56d3528b195540f91d6d2a75096ceff7", # enjoy the free API key to my local openai server (on my laptop)
)

from sympy import sympify

@myagent.tool # this is just langchain_core.tool reexported
def evaluate_sympy(expr: str):
    """Evaluates a sympy expression."""
    expr = sympify(expr)
    return str(expr.subs("x", 5))


messages = [
    myagent.UserMessage("What is 10293 ^ 2 + 108.")
]

myagent.run_agent(
    lm=client,
    messages=messages,
    model="unsloth/Qwen3.5-9B-GGUF",
    tools=[evaluate_sympy],
    callbacks=myagent.StreamingPrintCallback(),

    # args recommended for qwen
    temperature=0.6,
    top_p=0.95,
    presence_penalty=1.0,
    streaming=True,
    extra_body={
        "top_k": 20,
        "repetition_penalty": 1.0
    },
)
```

So whats the point. The point is that StreamingPrintCallback prints as the model writes tool calls in streaming mode. Langchain cant do that (or I havent found how).

```
── Step 0 ──

Thinking...The user is asking me to evaluate 10293^2 + 108. This is a simple arithmetic calculation that I can perform using the evaluate_sympy function.

Let me write this as a sympy expression: "10293^2 + 108"


  ⚡ evaluate_sympy({"expr":"10293^2 + 108"})
  → 105945957

── Step 1 ──

Thinking...The user wants to calculate the value of $10293^2 + 108$. I have already used the `evaluate_sympy` tool to get the result, which is 105945957. I can now provide the final answer directly.
The result of $10293^2 + 108$ is **105,945,957**.
```

its also simpler to set up than langchain in my humble opinion

