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

assigner_example = """
    {
        "person_0": "xx_job_name",
        "person_1": "yy_job_name",
        "person_2": "zz_job_name",
    }
"""

class Boss(BaseAgent):

    def __init__(self, name, llm):
        self.instruction = "Your are a assigner.You need to assign the job to the player based on the chat history.\
            Output example: ```\n{}\n```. \
            Note that you should only response the json string without any other characters.\
        ".format(assigner_example)
        super().__init__(name, self.instruction, llm, [])

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
