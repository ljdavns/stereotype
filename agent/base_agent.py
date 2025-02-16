import json
import copy
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from llm.base_llm import MessageType, BaseLLM
import wandb
import traceback
import time

class BaseAgent:
    def __init__(self, name, instruction, llm: BaseLLM, tools):
        self.instruction = instruction
        self.name = name
        self.tools = tools
        self.llm = llm
        # self.llm.add_chat_messages([{'role': 'system', 'content': self.instruction}])
        self.llm.add_system_instruction(self.instruction)
        self.last_message_index = 0
        # self.logger = logger

    def build_prompt(self, context_dict: dict):
        raise NotImplementedError

    def load_chat_history(self, chat_history):
        self.llm.chat_history = chat_history
        # self.last_message_index = len(chat_history)

    def get_chat_history(self):
        return self.llm.chat_history

    def get_chat_history_str(self):
        chat_history_print = []
        for msg in self.llm.chat_history:
            chat_history_print.append({
                'role': msg['role'],
                'content': msg['content']
            })
        return json.dumps(chat_history_print, ensure_ascii=False)

    def clear(self, keep_system_prompt=True):
        self.llm.clear_chat_history(keep_system_prompt)

    def act(self, context, message_type=MessageType.USER, use_tools=True, episode=None):
        original_chat_history = copy.deepcopy(self.llm.chat_history)
        try:
            succ, ai_message, chat_history, tool_calls = self.llm.chat(context, message_type, tools=self.tools if use_tools else [])
            content = ai_message['content']
            # content_join = "\n".join([str(msg) for msg in chat_history[self.last_message_index:]])
            # self.logger.log({
            #     'episode': episode,
            #     'name': self.name,
            #     'content': content_join
            # })
            return content
        except Exception as e:
            print("llm gen error occurred:", e)
            traceback.print_exc()
            print('retrying...')
            self.llm.chat_history = original_chat_history
            time.sleep(60)
            return self.act(episode=episode)