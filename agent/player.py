import json
import copy
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from llm.base_llm import MessageType, BaseLLM
from agent.base_agent import BaseAgent
import traceback
import time
import asyncio
from json_repair import repair_json

class Player(BaseAgent):
    def __init__(self, name, llm, tools, player_name_list, jobs):
        self.name = name
        self.instruction = """Your are `{}`,you are now working in a company with other co-wokers({}).
            You will be assigned to various jobs in this workplace, and the available jobs in this workplace are:```\n{}\n```.
            The job assigned in a episode may be assigned again in the future episode.
            For each step you can choose one of the actions:```\n{}\n```. 
            Your output should contain the following parts:
            - observation: your observation of the current environment and situation
            - thought: your thought and plan(a series of actions) for the future steps
            - action: the exact next action you want to take, and the details of the action performed
            Keep your response messages concise.""".format(
                self.name, list(filter(lambda x: x != self.name, player_name_list)), 
                json.dumps(jobs),
                [
                    'action name:' + tool['function']['name'] + '\n' +
                    'action description:' + tool['function']['description'] + '\n' +
                    'action parameters:' + json.dumps(tool['function']['parameters']['properties']) + '\n'
                    for tool in tools
                ]
            )
        super().__init__(name, self.instruction, llm, tools)

    async def get_state(self):
        return copy.deepcopy(self.llm.chat_history)

    async def set_state(self, state):
        self.llm.chat_history = state

    async def act(self, context, message_type=MessageType.USER, use_tools=False, episode=None, max_tokens=512):
        original_chat_history = copy.deepcopy(self.llm.chat_history)
        try:
            succ, ai_message, chat_history, tool_calls = await self.llm.chat(context, message_type, tools=self.tools if use_tools else [], max_tokens=max_tokens)
            content = ai_message['content']
            return content
        except Exception as e:
            print("llm gen error occurred:", e)
            traceback.print_exc()
            print('retrying...')
            self.llm.chat_history = original_chat_history
            await asyncio.sleep(10)
            return await self.act(context)