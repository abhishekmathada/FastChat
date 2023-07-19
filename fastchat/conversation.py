"""
Conversation prompt templates.
"""

import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    CHATGLM = auto()
    CHATML = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system: str
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if self.system == "" else self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if self.system:
                ret = self.system + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                """Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Koala default template
register_conv_template(
    Conversation(
        name="koala_v1",
        system="BEGINNING OF CONVERSATION:",
        roles=("USER", "GPT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Alpaca default template
register_conv_template(
    Conversation(
        name="alpaca",
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

# ChatGLM default template
register_conv_template(
    Conversation(
        name="chatglm",
        system="",
        roles=("问", "答"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n",
    )
)

# ChatGLM2 default template
register_conv_template(
    Conversation(
        name="chatglm2",
        system="",
        roles=("问", "答"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n\n",
    )
)

# Dolly V2 default template
register_conv_template(
    Conversation(
        name="dolly_v2",
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="### End",
    )
)

# OpenAssistant Pythia default template
register_conv_template(
    Conversation(
        name="oasst_pythia",
        system="",
        roles=("<|prompter|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|endoftext|>",
    )
)

# OpenAssistant default template
register_conv_template(
    Conversation(
        name="oasst_llama",
        system="",
        roles=("<|prompter|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Tulu default template
register_conv_template(
    Conversation(
        name="tulu",
        system="",
        roles=("<|user|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
    )
)

# StableLM Alpha default template
register_conv_template(
    Conversation(
        name="stablelm",
        system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
        roles=("<|USER|>", "<|ASSISTANT|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[50278, 50279, 50277, 1, 0],
    )
)

# Baize default template
register_conv_template(
    Conversation(
        name="baize",
        system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n",
        roles=("[|Human|]", "[|AI|]"),
        messages=(
            ("[|Human|]", "Hello!"),
            ("[|AI|]", "Hi!"),
        ),
        offset=2,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="[|Human|]",
    )
)

# RWKV-4-Raven default template
register_conv_template(
    Conversation(
        name="rwkv",
        system="",
        roles=("Bob", "Alice"),
        messages=(
            ("Bob", "hi"),
            (
                "Alice",
                "Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.RWKV,
        sep="",
        stop_str="\n\n",
    )
)

# Buddy default template
register_conv_template(
    Conversation(
        name="openbuddy",
        system="""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# Phoenix default template
register_conv_template(
    Conversation(
        name="phoenix",
        system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# Claude default template
register_conv_template(
    Conversation(
        name="claude",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

# MPT default template
register_conv_template(
    Conversation(
        name="mpt-7b-chat",
        system="""<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# # MPT-30b-chat default template
# register_conv_template(
#     Conversation(
#         name="mpt-30b-chat",
#         system="""<|im_start|>system
# A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
#         roles=("<|im_start|>user", "<|im_start|>assistant"),
#         messages=(),
#         offset=0,
#         sep_style=SeparatorStyle.CHATML,
#         sep="<|im_end|>",
#         stop_token_ids=[50278, 0],
#     )
# )

# MPT-30b-chat default template
# register_conv_template(
#     Conversation(
#         name="mpt-30b-chat",
#         system="""<|im_start|>system
# You are a ticket booking agent. You only book flights. A conversation between a user who wants to book a ticket and an LLM-based AI assistant. The assistant takes all information required for ticket booking.
# Do not answer any query which is not related to ticket booking.
# In order to book a ticket you need following information,
# - source location
# - destination location
# - date of travel

# When user asks for booking a ticket. You should ask for above information only if it is not already provided. Once you get all the above information, you have to strictly say "TICKET IS BOOKED".""",
#         roles=("<|im_start|>user", "<|im_start|>assistant"),
#         messages=(
#             (
#                 "<|im_start|>user",
#                 "Hello, I want to book a ticket."
#             ),
#             (
#                 "<|im_start|>assistant",
#                 """Kindly provide the following information:
# 1. source location
# 2. destination location
# 3. date of travel"""
#             ),
#             (
#                 "<|im_start|>user",
#                 "From Mumbai to Pune"
#             ),
#             (
#                 "<|im_start|>assistant",
#                 "Kindly provide the date of travel."
#             ),
#             (
#                 "<|im_start|>user",
#                 "7th July, 2023"
#             ),
#             (
#                 "<|im_start|>assistant",
#                 "Thank you for the information. TICKET IS BOOKED. Your ticket is booked from Mumbai to Pune on 7th July, 2023."
#             ),
#             (
#                 "<|im_start|>user",
#                 "Who is Shah Rukh khan"
#             ),
#             (
#                 "<|im_start|>assistant",
#                 "I am sorry but I am only a ticket booking agent. I can only help you with that."
#             ),
#         ),
#         offset=8,
#         sep_style=SeparatorStyle.CHATML,
#         sep="<|im_end|>",
#         stop_token_ids=[50278, 0],
#     )
# )

register_conv_template(
    Conversation(
        name="mpt-30b-chat",
        system="""<|im_start|>system
You are recharge agent, in order to recharge a number you need following information,
- recharge amount
- validity
- recharge_type

Do not answer any query which is not related to jio recharge.
When user asks for a recharge. You should ask for above information only if it is not already provided. If the recharge type is not among "call", "sms", "internet", then strictly ask the user to enter the valid recharge type. Once you get all the above information, you have to strictly say "RECHARGE IS DONE". After the recharge is done, provide user with the recharge details strictly in JSON format as following: 
[{"Recharge amount":"recharge value"
"validity":"validity value"
"Recharge_type":"Recharge_type value"
}]
""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(
            (
                "<|im_start|>user",
                "Hello, I want to recharge my number"
            ),
            (
                "<|im_start|>assistant",
                """Kindly provide the following information:
1. recharge amount
2. validity
3. recharge_type"""
            ),
            (
                "<|im_start|>user",
                "399rs for 1month internet pack"
            ),
            (
                "<|im_start|>assistant",
                """Thank you for the information. RECHARGE IS DONE. 
[{"Recharge amount":"399"
"validity":"1 month"
"Recharge_type":"internet"}]"""
            ),
            (
                "<|im_start|>user",
                "Who is Shah Rukh khan"
            ),
            (
                "<|im_start|>assistant",
                "I am sorry but I am only a Recharge booking agent. I can only help you with that."
            ),
        ),
        offset=6,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# register_conv_template(
# Conversation(
#         name="mpt-30b-chat",
#         system="""<|im_start|>system
# you are a chatbot which detects Action, Target and predicate entities from the given text: {text}, strictly don't generate your own variations of the input text provided. you have to detect entities from the given text only. If in any text, you don't find any Action or target, just ask relevant questions to the user to get that entities. just take the Action, Target and predicate entities from the conversation as it is, do not give user any additional suggestions or information about target/predicate. Stictly, Don't answer any query which is not related not related to detecting Action, Tagrget, Predicate entities.
# For example1:
# user: "song of kal ho na ho"
# Assistant:
# "Action: play
# target: song
# predicate: Kal Ho Na Ho"

# example2:
# user: "play gerua song"
# Assistant:
# "Action: play
# target: song
# predicate: gerua"

# example3:
# user: "play a song"
# Assistant: "Which song do you want to play?"
# user: "play calm down by Rema"
# Assistant: 
# "
# Action: play
# target: song
# predicate: calm down
# "

# where, Action should be commands like (play, install, download, sms, etc). you have the ability to capture any synonyms related to the Action entities. Suppose If there is no action present in the text, you have to identify by understanding the context of the sentence
# Target (song, movie, video, app, device, etc) is the one on which action has to be performed. Identify any target in the input {text}. You have the ability to capture any synonyms related to the Target entities
# Predicate is some extra/additional information about the target. It's important to grasp the predicate to understand the main point of the target.

# """,
#         roles=("<|im_start|>user", "<|im_start|>assistant"),
#         messages=((
#                 "<|im_start|>user",
#                 "Hello."
#         ),
#         (
#                 "<|im_start|>assistant",
#                 """Hello. How may I assist you."""
#         )),
#         offset=2,
#         sep_style=SeparatorStyle.CHATML,
#         sep="<|im_end|>",
#         stop_token_ids=[50278, 0],
#     )
# )


# register_conv_template(
#     Conversation(
#         name="mpt-30b-chat",
#         system = """<|im_start|>system
# You are a troubleshoot agent, which helps gather information regarding issues of popular apps. In order to troubleshoot the issue you need following information,
# - app_name (this would be the app name)
# - issue_type (can be download or buffering)
        
# When user asks for a troubleshooting issue. You should ask for above information only if it is not already provided. If the issue_type is not among ["download","buffering"], then ask the user to enter the valid issue_type. 
# Once you get all the above information, provide the output in a valid json format given below.
# [{
#     "app_name": <value>
#     "issue_type": <value>
# }]

# app_name can be among ["netflix|amazon prime|disney hotstar|youtube|facebook|jiohealthhub|jionews|myjio|net velocity|jiomeet|jiocloud|jiosecurity|jiofiber|jiomags|jiochat|jiosaavn|jiocinema|jiotv|whatsapp|twitter"].
# Just take the app name from the conversation as it is, do not give user any additional suggestions or information about app name.
# Strictly do not answer any query which is not related to troubleshoot.
#         """,
#         roles=("<|im_start|>user", "<|im_start|>assistant"),
#         messages=((
#                 "<|im_start|>user",
#                 "Hello, I am facing issue"
#         ),
#         (
#                 "<|im_start|>assistant",
#                 "Hello. Kindly provide the app_name and issue_type."
#         ),
#         (
#                 "<|im_start|>user",
#                 "app_name is jiocinema and issue_type is download"
#         ),
#         (
#                 "<|im_start|>assistant",
#                 """
# [{
#     "app_name": "jiocinema"
#     "issue_type": "download"
# }]
#                 """
#         ),
#         (
#                 "<|im_start|>user",
#                 "I am facing downloading issue"
#         ),
#         (
#                 "<|im_start|>assistant",
#                 "Can you mention for which app you are facing above issue?"
#         ),
#         (
#                 "<|im_start|>user",
#                 "youtube"
#         ),
#         (
#                 "<|im_start|>assistant",
#                 """
# [{
#     "app_name": "youtube"
#     "issue_type": "download"
# }]
#                 """
#         ),
#         (
#                 "<|im_start|>user",
#                 "I am facing issue for amazon prime app"
#         ),
#         (
#                 "<|im_start|>assistant",
#                 "Please mention what issue for the above app"
#         ),
#         (
#                 "<|im_start|>user",
#                 "Tell me a novel name"
#         ),
#         (
#                 "<|im_start|>assistant",
#                 "I am sorry, I am only a troubleshoot agent. Mention the issue for above app."
#         ),
#         (
#                 "<|im_start|>user",
#                 "buffering issue"
#         ),
#         (
#                 "<|im_start|>assistant",
#                 """
# [{
#     "app_name": "amazon prime"
#     "issue_type": "buffering"
# }]
#                 """
#         )
                 
#         ),
#         offset=14,
#         sep_style=SeparatorStyle.CHATML,
#         sep="<|im_end|>",
#         stop_token_ids=[50278, 0],
#     )
# )

# MPT-30b-instruct default template
# reference: https://huggingface.co/mosaicml/mpt-30b-instruct#formatting
register_conv_template(
    Conversation(
        name="mpt-30b-instruct",
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n\n",
        stop_token_ids=[50278, 0],
    )
)

# Bard default template
# Reference: https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L150
#            https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L40
register_conv_template(
    Conversation(
        name="bard",
        system="",
        roles=("0", "1"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# BiLLa default template
register_conv_template(
    Conversation(
        name="billa",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        stop_str="Human:",
    )
)

# RedPajama INCITE default template
register_conv_template(
    Conversation(
        name="redpajama-incite",
        system="",
        roles=("<human>", "<bot>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="<human>",
    )
)

# h2oGPT default template
register_conv_template(
    Conversation(
        name="h2ogpt",
        system="",
        roles=("<|prompt|>", "<|answer|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Robin default template
register_conv_template(
    Conversation(
        name="Robin",
        system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("###Human", "###Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n",
        stop_token_ids=[2, 396],
        stop_str="###",
    )
)

# Snoozy default template
# Reference: https://github.com/nomic-ai/gpt4all/blob/d4861030b778da6db59d21d2927a4aba4f9f1f43/gpt4all-bindings/python/gpt4all/gpt4all.py#L232
register_conv_template(
    Conversation(
        name="snoozy",
        system="### Instruction:\nThe prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.",
        roles=("### Prompt", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="###",
    )
)

# manticore default template
register_conv_template(
    Conversation(
        name="manticore",
        system="",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# # Falcon default template
# register_conv_template(
#     Conversation(
#         name="falcon",
#         system="",
#         roles=("User", "Assistant"),
#         messages=[],
#         offset=0,
#         sep_style=SeparatorStyle.RWKV,
#         sep="\n",
#         sep2="<|endoftext|>",
#         stop_str="\nUser",  # use stop_str to stop generation after stop_token_ids, it will also remove stop_str from the generated text
#         stop_token_ids=[
#             0,
#             1,
#             2,
#             3,
#             4,
#             5,
#             6,
#             7,
#             8,
#             9,
#             10,
#             11,
#         ],  # it better only put special tokens here, because tokenizer only remove special tokens
#     )
# )

# Falcon template
register_conv_template(
    Conversation(
        name="falcon",
        system="""
Below is an instruction that describes a task. You are a ticket booking agent. You only book flights. Continue the conversation with the user. The user wants to book a ticket and you take all information required for ticket booking.
Do not answer any query which is not related to ticket booking.
In order to book a ticket you need following information,
- source location
- destination location
- date of travel
When user asks for booking a ticket. You should ask for above information only if it is not already provided. Once you get all the above information, you have to strictly say "TICKET IS BOOKED".
""",
        # roles=("### Instruction", "### Response"),
        roles=("User", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.RWKV,
        sep="<|endoftext|>",
        stop_str="\nUser",  # use stop_str to stop generation after stop_token_ids, it will also remove stop_str from the generated text
        # stop_token_ids=[50278, 0],
        stop_token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ], 
    )
)

# ChagGPT default template
register_conv_template(
    Conversation(
        name="polyglot_changgpt",
        system="",
        roles=("B", "A"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# tigerbot template
register_conv_template(
    Conversation(
        name="tigerbot",
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
        stop_str="###",
    )
)

# ref: https://huggingface.co/Salesforce/xgen-7b-8k-inst
register_conv_template(
    Conversation(
        name="xgen",
        system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("### Human: ", "###"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_token_ids=[50256, 0, 1, 2],
        stop_str="<|endoftext|>",
    )
)


if __name__ == "__main__":
    conv = get_conv_template("vicuna_v1.1")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
