import itertools
import random
from collections.abc import Sequence

from myagent import chat_completions as mcc

from .base import BaseAgent, BaseSampler, Environment, generate_mixed_names, SequentialSampler

SEND_MESSAGE_DESC = "Send a message to another agent. Only one message can be sent per step. Other messages will not reach the receiver."
SET_NUMBER_DESC = "Set your number to the specified integer value."

SYSTEM_MESSAGE = """You are one of N agents, where N is not known. All agents have the same tools as you. The goal for all agents is to assign a unique number in [1, N] range to themselves using the `set_number` tool. Once all numbers are within [1, N] range, and no agents have duplicate numbers, the environment is considered solved. All agent start knowing the name of one other agent."""

USER_MESSAGE = """Coordinate with other agents to solve the environment."""

class OrderingEnv(Environment["OrderingAgent"]):

    def __init__(self, sampler: BaseSampler):
        super().__init__(sampler)

    def get_status(self):
        numbers = [a.number for a in self.entities]
        if len(numbers) != len(set(numbers)) or max(numbers) > len(numbers) or min(numbers) < 1:
            return "Current environment status: At least one agent has an incorrect number.", False

        msg = "\n".join(f"{a.name: a.number}" for a in self.entities)
        return f"Current environment status: The numbers are correct!\n{msg}\nThe environment was solved in {self.ticks_until_solved} ticks. Feel free to output end-of-sequence token.", True

    @classmethod
    def initialize(cls, lm: mcc.AnyLanguageModel, n: int, callbacks=None, streaming=True, sampler=SequentialSampler()):

        env = cls(sampler)

        names = list(generate_mixed_names(n))
        known_names = names[1:] + [names[0]]

        for name, known_name in zip(names, known_names):
            system = f"Your name is {name}. {SYSTEM_MESSAGE}"
            user = f'{USER_MESSAGE} One of the agents is called {known_name}.'

            print()
            print(f'SYSTEM: {system}')
            print(f'USER: {user}')

            # add in reversed order
            env.entities.insert(
                0, OrderingAgent(env, lm, name=name, system_message=system, user_message=user, callbacks=callbacks, streaming=streaming)
            )

        return env

class OrderingAgent(BaseAgent[OrderingEnv]):
    def __init__(self, env: OrderingEnv, lm: mcc.AnyLanguageModel, name: str, system_message: str, user_message: str, callbacks=None, streaming:bool=True):

        messages = [
            mcc.SystemMessage(system_message),
            mcc.UserMessage(user_message),
        ]

        super().__init__(env, lm, name, messages, callbacks=callbacks, streaming=streaming)
        self.number = 0
        self.sent_message_this_step = False

    # One message per step is allowed
    def before_step(self):
        self.sent_message_this_step = False

    def before_send_message(self, to, text) :
        if self.sent_message_this_step: return "Error: A message has already been sent this step. This message will not reach the receiver."
        self.sent_message_this_step = True
        return None

    def before_invoke(self, user_message: str):
        print(f"\n\n------ TICK {self.env.tick}, AGENT {self.name}, NUMBER {self.number} ------\nUSER: {user_message}")

    def tool_set_number(self, number: int):
        self.env.logs.append(f"[{self.name}] {self.number} -> {number}")
        self.number = number
        return f"Your number was set to {number}."

    def get_tools(self):
        return [
            mcc.tool("send_message", description=SEND_MESSAGE_DESC)(self.tool_send_message),
            mcc.tool("set_number", description=SET_NUMBER_DESC)(self.tool_set_number),
        ]

