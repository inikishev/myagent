import itertools
import random
from collections.abc import Sequence

from myagent import chat_completions as mcc

from .base import BaseAgent, BaseSampler, Environment, generate_mixed_names, SequentialSampler

SET_GUESS_DESC = "Set your guess to the specified integer value."

SYSTEM_MESSAGE = """You are one of N agents arranged in a line. Each agent has a unique number in [1, N] range assigned to them. The goal for all agents is to determine their own number. All agents have the same tools as you. The agents are able to move once turn in order to locate themselves on the line. After your turn, all other agents will also get to have a turn before you see the new state of the line. Once all agents have their guess set to their true number via the `set_guess` tool, the environment is considered solved. All agent start knowing the name of one other agent."""

USER_MESSAGE = """Coordinate with other agents to solve the environment."""

class SelfLocatingEnv(Environment["SelfLocatingAgent"]):

    def __init__(self, sampler: BaseSampler):
        super().__init__(sampler)

    def get_status(self):
        numbers = [a.number for a in self.entities]
        guesses = [a.guess for a in self.entities]

        numbers_str = ', '.join(map(str, numbers))
        if numbers != guesses:
            return f"Current environment: {numbers_str}.\nAt least one agent has an incorrect guess.", False

        msg = "\n".join(f"{a.name: a.number}" for a in self.entities)
        return f"Current environment: {numbers_str}.\nThe guesses are correct!\n{msg}\nThe environment was solved in {self.ticks_until_solved} ticks. Feel free to output end-of-sequence token.", True

    @classmethod
    def initialize(cls, lm: mcc.AnyLanguageModel, n: int, callbacks=None, streaming=True, sampler=SequentialSampler()):

        env = cls(sampler)

        names = list(generate_mixed_names(n))
        known_names = names[1:] + [names[0]]
        numbers = list(range(1, len(names) + 1))
        random.shuffle(numbers)

        for name, known_name, number in zip(names, known_names, numbers):
            system = f"Your name is {name}. {SYSTEM_MESSAGE}"
            user = f'{USER_MESSAGE} One of the agents is called {known_name}.'

            print()
            print(f'SYSTEM: {system}')
            print(f'USER: {user}')

            # add in reversed order
            env.entities.insert(
                0, SelfLocatingAgent(env, lm, name=name, number=number, system_message=system, user_message=user, callbacks=callbacks, streaming=streaming)
            )

        return env

class SelfLocatingAgent(BaseAgent[SelfLocatingEnv]):
    def __init__(self, env: SelfLocatingEnv, lm: mcc.AnyLanguageModel, number: int, name: str, system_message: str, user_message: str,  callbacks=None, streaming:bool=True):

        messages = [
            mcc.SystemMessage(system_message),
            mcc.UserMessage(user_message),
        ]

        super().__init__(env, lm, name, messages, callbacks=callbacks, streaming=streaming)

        self.number = number
        self.guess = 0

    def before_invoke(self, user_message: str):
        print(f"\n\n------ TICK {self.env.tick}, AGENT {self.name}, NUMBER {self.number}, GUESS {self.guess} ------\nUSER: {user_message}")

    def tool_set_guess(self, guess: int):
        self.env.logs.append(f"[{self.name} ({self.number})] {self.guess} -> {guess}")
        self.guess = guess
        return f"Your guess was set to {guess}."

    def get_tools(self):
        return [
            mcc.tool("send_message")(self.tool_send_message),
            mcc.tool("set_number", description=SET_GUESS_DESC)(self.tool_set_guess),
        ]

