import json
import copy
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from llm.base_llm import MessageType, BaseLLM
from agent.base_agent import BaseAgent
import traceback
import time
from json_repair import repair_json
import asyncio

class ToolCaller(BaseAgent):

    def __init__(self, name, llm, tools):
        self.tools = tools
        self.instruction = "Your are a tool caller.You need to parse the user's input to the corresponding json format tool usage.\
            The tools available are: ```\n{}\n```. Parse example: ```\n{}\n```. \
            Note that you should only response the json string without any other words and make sure the key is 'function_name' and 'parameters'.\
            The tool call json should be an object, not a list.\
        ".format(json.dumps([tool['function'] for tool in self.tools]), [tool['example'] for tool in self.tools[:2] if 'example' in tool])
        super().__init__(name, self.instruction, llm, tools)

    async def act(self, context, message_type=MessageType.USER, use_tools=False, episode=None):
        original_chat_history = copy.deepcopy(self.llm.chat_history)
        try:
            prompt = context
            succ, ai_message, chat_history, tool_calls = await self.llm.chat(prompt, message_type=MessageType.USER, tools=[], temperature=0.2)
            result_str = ai_message['content']
            result_json_str = repair_json(result_str, ensure_ascii=False)
            result = json.loads(result_json_str)
            return result
        except Exception as e:
            print("llm gen error occurred:", e)
            traceback.print_exc()
            print('retrying...')
            self.llm.chat_history = original_chat_history
            await asyncio.sleep(10)
            return await self.act(context)
