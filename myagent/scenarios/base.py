import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Annotated

from faker import Faker

from myagent import chat_completions as mcc


class BaseEntity[ENV: "Environment"](ABC):
    def __init__(self, env: ENV, name: str, ):
        self.env: ENV = env
        self.name = name
        self.n_steps = 0

    def receive_notification(self, text: str):
        pass

    def receive_message(self, sender: "BaseEntity", text: str):
        pass

    def on_tick(self):
        """runs every tick"""

    def before_step(self):
        pass

    def step(self) -> bool:
        """Returns whether any action was taken."""
        return False

def _normalize_string(s: str):
    return s.lower().strip()

class EntityNotFound(Exception): pass


class BaseSampler(ABC):
    @abstractmethod
    def sample[T: BaseEntity](self, entities: list[T]) -> list[T]:
        pass

class RandomSampler(BaseSampler):
    def sample(self, entities):
        return [random.choice(entities)]

class SequentialSampler(BaseSampler):
    def sample(self, entities):
        return entities

class PermutationSampler(BaseSampler):
    def sample(self, entities):
        return random.sample(entities, k=len(entities))

# env and agent have a generic for the other, this is to get better IDE hints
class Environment[ENTITY_TYPE: BaseEntity](ABC):
    def __init__(self, sampler: BaseSampler):
        self.entities: list[ENTITY_TYPE] = []
        self.tick = 0
        self.sampler = sampler

        self.pending_entities: list[ENTITY_TYPE] = []
        self.ticks_until_solved = None
        self.logs: list[str] = []

    def get_entity_by_name(self, name: str):
        for entity in self.entities:
            if _normalize_string(name) == _normalize_string(entity.name):
                return entity
        raise EntityNotFound(f"No entity named {name}. Available names: {[e.name for e in self.entities]}")

    @abstractmethod
    def get_status(self) -> tuple[str,bool]:
        """Returns current status and success boolean, e.g. is the env solved or not."""

    def step(self) -> bool:
        if self.ticks_until_solved is None:
            _, solved = self.get_status()
            if solved: self.ticks_until_solved = self.tick

        if not self.pending_entities:
            self.pending_entities.extend(self.sampler.sample(self.entities))

        cur_entity = self.pending_entities.pop(0)
        stepped = cur_entity.step()

        for entity in self.entities:
            entity.on_tick()

        self.tick += 1
        return stepped

    def run(self, max_steps: int):

        for _ in range(max_steps):
            self.step()


class Notification:
    def __init__(self, receiver: BaseEntity, text: str, tick: int):
        self.receiver = receiver
        self.text = text
        self.tick = tick

    def to_string(self):
        return f"{self.text}"

class Message(Notification):
    def __init__(self, sender: BaseEntity, receiver: BaseEntity, text: str, tick: int):
        super().__init__(receiver, text, tick)
        self.sender = sender

    def to_string(self):
        return f"You received a message:\n[FROM: {self.sender.name}, TO: {self.receiver.name}] {self.text}"

class BaseAgent[ENV_TYPE: Environment](BaseEntity, ABC):
    env: ENV_TYPE

    def __init__(self, env: ENV_TYPE, lm: mcc.AnyLanguageModel, name: str, messages: Sequence[mcc.AnyMessage], callbacks=None, streaming:bool=True):
        super().__init__(env, name)
        self.lm = mcc.get_lm(lm)
        self.messages = list(messages)
        self.unread_notifications: list[Notification] = []

        self.callbacks = callbacks
        self.streaming = streaming

    def before_send_message(self, to: str, text: str) -> str | None:
        pass

    def tool_send_message(
        self,
        to: Annotated[str, "Name of the agent to send message to."],
        text: Annotated[str, "Message to send."],
    ) -> str:
        """Send a message to another agent."""
        msg = self.before_send_message(to, text) # pylint:disable=E1111
        if msg:
            return msg

        try:
            receiver = self.env.get_entity_by_name(to)
            self.env.logs.append(f"[{self.name} -> {receiver.name}] {text}")
            receiver.receive_message(sender=self, text=text)
            return "Message has been sent."

        except EntityNotFound:
            return f"Error: agent with name `{to}` doesn't exist."

    def receive_notification(self, text: str):
        self.unread_notifications.append(Notification(receiver=self, text=text, tick=self.env.tick))

    def receive_message(self, sender, text):
        self.unread_notifications.append(Message(sender=sender, receiver=self, text=text, tick=self.env.tick))

    def get_unread_string(self):
        """Returns a string with unread events to give to the agent. Only includes events themselves, no header.
        Don't forget to clean `unread_notifications` afterwards."""
        strings = [f'{i}. {notif.to_string()}' for i,notif in enumerate(self.unread_notifications, start=1)]
        return '\n\n'.join(strings)

    @abstractmethod
    def get_tools(self) -> list[mcc.BaseTool]:
        """Returns tools of the agent."""

    def get_status(self) -> tuple[str,bool]:
        """Returns current status and success boolean, e.g. is the env solved or not. By default uses env status."""
        return self.env.get_status()

    def before_invoke(self, user_message: str):
        print(f"\n\n------ TICK {self.env.tick}, AGENT {self.name} ------\nUSER: {user_message}")

    def step(self):

        status, success = self.get_status()

        if (self.unread_notifications) or (not success):

            self.before_step()

            # Construct a status message
            if self.n_steps != 0 or self.unread_notifications:

                msg = f'<SYSTEM MESSAGE>\n{status}'
                if self.unread_notifications:
                    msg = f"{msg}\n\nYou have {len(self.unread_notifications)} new notification(s):\n\n{self.get_unread_string()}"
                    self.unread_notifications.clear()
                else:
                    msg = f"{msg}\n\nNothing happened since your last turn."

                self.messages.append(mcc.UserMessage(msg))
                self.before_invoke(msg)

            else:
                assert len(self.unread_notifications) == 0
                self.before_invoke("(first step)")

            # Invoke the language model
            tools=self.get_tools()
            response = mcc.agent_step(
                lm=self.lm,
                messages=self.messages,
                tools=tools,
                streaming=self.streaming,
                callbacks=self.callbacks,
                tool_choice="auto" if success else "required"
            )
            self.messages.append(response)

            tool_calls = response.invoke_tools(tools, callbacks=self.callbacks)
            self.messages.extend(tool_calls)

            self.n_steps += 1
            return True

        return False



def generate_first_names(n: int):
    names = set()
    faker = Faker()
    while len(names) < n:
        names.add(faker.name().split(' ')[0])
    return sorted(names)

def generate_greek_letter_names(n: int):
    choices = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
    ]
    return sorted(random.sample(choices, k=n))

def generate_word_based_names(n: int):
    choices = [
        "Cactus", "Cat", "Mouse", "Lumberjack", "Sponge", "Apple", "Cherry", "Banana", "Strawberry", "Speaker", "Micro", "Mini", "Nano", "Pico", "Melody", "Pineapple", "Spruce", "Oak", "Mixer", "Anti", "Macro", "Atom", "Nut", "Berry", "Pixel", "Forest", "River", "Cloud", "Storm", "Shield", "Flow", "Zero", "Triangle", "Star", "Sky", "Fire", "Water", "Autumn", "Winter", "Summer", "Speedy", "Speedrunner", "Shadow", "Fisherman", "Camper", "Spring", "Tensor", "Vector", "Manifold", "Farmer", "Dino", "Sugar", "Silver", "Leaf", "Flower", "Blade", "Bee", "Red", "Green", "Blue", "Yellow", "Orange", "Lime", "Lemon", "Ant", "Ivy", "Leaf", "Mosquito", "Fox", "Bat",
    ]
    assert len(choices) == len(set(choices))
    return sorted(random.sample(choices, k=n))


def generate_mixed_names(n: int):
    nd = n // 3
    names = generate_first_names(nd) + generate_greek_letter_names(nd)
    names.extend(generate_word_based_names(n-len(names)))
    return names
