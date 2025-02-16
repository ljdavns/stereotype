from enum import Enum

class MessageType(Enum):
    SYSTEM = "system"
    USER = "user"
    AI = "assistant"

class BaseLLM:

    def __init__(self, model_name):
        self.model_name = model_name
        self.chat_history = []

    def add_chat_messages(self, messages: list):
        self.chat_history += messages
    
    def add_system_instruction(self, instruction):
        raise NotImplementedError

    def clear_chat_history(self, keep_system_prompt=True):
        self.chat_history = self.chat_history[:1] if keep_system_prompt else []
    
    def chat(self, new_message, message_type: MessageType, chat_history: list = None, **kwargs):
        raise NotImplementedError
    